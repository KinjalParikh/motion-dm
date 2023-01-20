import torch
from utils.misc_utils import get_device


class GaussianDiffusion:
    def __init__(self, num_time_steps=1000):
        self.num_time_steps = num_time_steps
        self.beta = self.get_betas()

        alpha = torch.ones_like(self.beta) - self.beta
        self.alpha_cum_prod = torch.cumprod(alpha, dim=0)
        self.alpha_cum_prod_prev = torch.cat((torch.tensor([1]), self.alpha_cum_prod[:-1]), dim=0)

        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

        self.beta_post = torch.divide((1 - self.alpha_cum_prod_prev), (1 - self.alpha_cum_prod)) * self.beta

        self.posterior_mean_coef_x0 = torch.divide(torch.sqrt(self.alpha_cum_prod_prev), (1 - self.alpha_cum_prod)) \
                                      * self.beta
        self.posterior_mean_coef_xt = torch.divide((1 - self.alpha_cum_prod_prev), (1 - self.alpha_cum_prod)) \
                                      * self.sqrt_alpha_cum_prod

    def q_sample(self, x_start, time_step, noise=None):
        """
        Adds noise to x_start time_step times
        :param x_start: x_start from the data set
        :param time_step: time_steps for each sample in data set #TODO t - 1 expected
        :param noise:
        :return: x_t: result of applying noising kernel on x_start time_step times

        Used during training
        """
        if not noise:
            noise = torch.rand_like(x_start)
        noise = noise.to(get_device())
        assert noise.shape == x_start.shape

        mean_coef = _extract_into_tensor(self.sqrt_alpha_cum_prod, time_step, x_start.shape)
        var = _extract_into_tensor(self.sqrt_one_minus_alpha_cum_prod, time_step, x_start.shape)

        x_t = mean_coef * x_start + var * noise
        return x_t

    def q_posterior_sample(self, x_start, x_t, time_step, noise=None):
        """
        Used during generation. Predicted x_start is noised back to x_{t-1} given x_t.
        """
        if not noise:
            noise = torch.rand_like(x_start)
        noise = noise.to(x_start.device)
        assert noise.shape == x_start.shape

        mean_coef_x0 = _extract_into_tensor(self.posterior_mean_coef_x0, time_step, x_start.shape)
        mean_coef_xt = _extract_into_tensor(self.posterior_mean_coef_xt, time_step, x_t.shape)

        var = _extract_into_tensor(self.beta_post, time_step, noise.shape)

        assert x_start.shape == x_t.shape

        x_tprev = mean_coef_x0 * x_start + mean_coef_xt * x_t + var * noise
        return x_tprev

    def p_sample(self, x_t, model, time_steps):
        """
        Uses MDM model to predict x_0 given x_t and then calculates q(x_{t-1}|x_0, x_t)
        """
        x_start_pred = model(x_t, time_steps)
        x_prev = self.q_posterior_sample(x_start_pred, x_t, time_steps)
        return x_prev

    def p_sample_loop(self, model, num_samples=3, noise=None, progress=True):
        """
        Used for generation.
        :param model: the MDM model
        :param num_samples: number of motions to be generated
        :param noise:
        :param progress: if true, tqdm progress bar will be displayed
        :return: #num_samples generated motions
        """
        if noise is not None:
            x_t = noise
        else:
            shape = (num_samples, model.njoints, model.nfeats, 40)
            #TODO: change  to model.nframes and create nframes attribute in model
            x_t = torch.randn(shape)
        x_t = x_t.to(get_device())

        time_steps_dec = list(range(self.num_time_steps)[::-1])

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(time_steps_dec)
        else:
            indices = time_steps_dec

        for i in indices:
            t = torch.tensor([i] * num_samples, device=get_device())
            with torch.no_grad():
                prev_x = self.p_sample(x_t, model, t)
                x_t = prev_x

        return x_t

    def get_betas(self):
        """
        :return: beta for time steps 0 through T. Linear schedule applied
        """
        beta = torch.linspace(0.0001, 0.02, self.num_time_steps, dtype=torch.float32)
        return beta


def _extract_into_tensor(arr, indices, broadcast_shape):
    """
    selects indices from arr and broadcasts it to required shape
    :param arr:
    :param indices: list of indices to be selected from arr. (list of time steps)
    :param broadcast_shape: target shape. (shape of x_start)
    :return:
    """
    arr_device = arr.to(get_device())
    selected = arr_device[indices]
    # add dimensions [1, 2, 3] -> [[1], [2], [3]]
    while len(broadcast_shape) > len(selected.shape):
        selected = selected[..., None]
    # expand along the dimensions eg, [[1, 1], [2, 2], [3, 3]]
    selected = selected.expand(broadcast_shape)
    return selected