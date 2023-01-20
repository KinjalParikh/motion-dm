import os
import torch
from utils import misc_utils, parser_utils, paramUtil
from utils.model_utils import create_model_and_diffusion
import numpy as np
import shutil


def main():

    args = parser_utils.generate_args()

    misc_utils.fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)

    args.batch_size = args.num_samples  # Sampling a single batch from the test set, with exactly args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location=misc_utils.get_device())
    model.load_state_dict(state_dict, strict=False)

    model.to(misc_utils.get_device())
    model.eval()  # disable random masking

    samples = diffusion.p_sample_loop(model, num_samples=args.num_samples)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, samples.cpu().numpy())


if __name__ == "__main__":
    main()