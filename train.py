# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.misc_utils import fixseed, get_device
from utils.parser_utils import train_args
from training_loop import TrainLoop
from data import get_dataset_loader
from utils.model_utils import create_model_and_diffusion
# from utils.log_platform import TensorboardPlatform, NoPlatform  # required for the eval operation


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # train_platform_type = eval(args.train_platform_type)
    # train_platform = train_platform_type(args.save_dir)
    # train_platform.report_args(args, name='Args')

    print("creating data loader...")
    data = get_dataset_loader(batch_size=args.batch_size)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    model.to(get_device())
    model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")

    TrainLoop(args, model, diffusion, data).run_loop()


if __name__ == "__main__":
    main()
