from model import MDM
import diffusion as df


def create_model_and_diffusion(args):
    model = MDM(**get_model_args(args))
    diffusion = df.GaussianDiffusion(num_time_steps=args.diffusion_steps)
    return model, diffusion


def get_model_args(args):
    # human act data set, xyz positions
    njoints = 24
    nfeats = 3

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu"}
