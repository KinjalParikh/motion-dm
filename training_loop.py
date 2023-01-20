import os
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter
from utils import log_utils as logger
from utils.misc_utils import get_device
from tqdm import tqdm
from losses import losses

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from torch.profiler import profile, record_function, ProfilerActivity


class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        # self.dataset = args.dataset
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.device = get_device()

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.tensorboard_writer = SummaryWriter(log_dir=self.save_dir)
        # self.tensorboard_writer.add_graph(model, [torch.rand(64, 24, 3, 40), torch.randint(1, 1000, [64])])

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(torch.load(resume_checkpoint, map_location="cpu"))

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(torch.load(opt_checkpoint, map_location="cpu"))

    def run_loop(self):
        torch.autograd.set_detect_anomaly(True)
        try:
            for epoch in range(self.num_epochs):
                print(f'Starting epoch {epoch}')
                for motion in tqdm(self.data):
                    if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                        break

                    motion = motion.to(torch.float32).to(self.device)
                    self.run_step(motion)
                    if self.step % self.log_interval == 0:
                        for k,v in logger.get_current().name2val.items():
                            if k == 'loss':
                                print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                    if self.step % self.save_interval == 0:
                        self.save()

                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                            return
                    self.step += 1
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
            # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0:
                self.save()
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
        except KeyboardInterrupt:
            self.save()
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()

    def run_step(self, batch):
        self.forward_backward(batch)
        # self.opt.step()
        self._anneal_lr()

    def forward_backward(self, batch):
        self.opt.zero_grad()
        t = torch.from_numpy(
            np.random.choice(self.diffusion.num_time_steps, size=(batch.shape[0]))
        ).long().to(self.device)

        noised_motion = self.diffusion.q_sample(batch, t)
        motion_pred = self.model(noised_motion, t)

        losses_dict = losses(motion_pred, batch)
        loss = losses_dict["loss"].mean()

        self.tensorboard_writer.add_scalar('loss', loss, global_step=self.step)
        if self.step % self.log_interval == 0:
            print("Step: ", self.step, "; Loss: ", loss)
        loss.backward()
        # plot_grad_flow(self.model.named_parameters())
        self.opt.step()
        # print(list(self.model.parameters())[0].grad)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        param_list = list(self.model.parameters())
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = param_list[i]
        logger.log(f"saving model...")
        filename = self.ckpt_file_name()
        with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
            torch.save(state_dict, f)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def find_resume_checkpoint():
    pass


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    credit: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])