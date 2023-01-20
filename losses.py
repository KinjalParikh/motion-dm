import torch


def losses(motion_pred, motion, lambda_pos=0.65, lambda_fc=0.1, lambda_vel=0.25):

    assert motion_pred.shape == motion.shape
    # shape: bs, njoints, nfeats, nframes

    loss_pos = mean_flat(torch.pow((motion - motion_pred), 2))

    loss_vel = mean_flat(torch.pow((vel(motion) - vel(motion_pred)), 2))

    foot_vels = torch.pow(torch.sum(torch.pow(vel(motion[:, [7, 8, 10, 11], :, :]), 2), dim=2), 0.5)
    foot_vels_pred = torch.pow(torch.sum(torch.pow(vel(motion_pred[:, [7, 8, 10, 11], :, :]), 2), dim=2), 0.5)

    # foot_vels = torch.linalg.norm(vel(motion[:, [7, 8, 10, 11], :, :]), dim=2)    #shape: bs X 4 X nframes now
    # foot_vels_pred = torch.linalg.norm(vel(motion_pred[:, [7, 8, 10, 11], :, :]), dim=2)

    fc_mask = foot_vels <= 0.01
    foot_vels_pred[~fc_mask] = 0
    loss_fc = sum_flat(torch.pow(foot_vels_pred, 2))/motion.shape[3]

    loss = lambda_pos*loss_pos + lambda_fc*loss_fc + lambda_vel*loss_vel
    terms = {
        "loss": loss,
        "pos": loss_pos,
        "vel": loss_vel,
        "fc": loss_fc
    }
    return terms


def sum_flat(a):
    """
        calculates mean over all dimensions except the first one
        :param a: any tensor
        :return: tensor of length a.shape[0], containing mean values
        """
    return torch.sum(a, list(range(1, len(a.shape))))


def mean_flat(a):
    """
    calculates mean over all dimensions except the first one
    :param a: any tensor
    :return: tensor of length a.shape[0], containing mean values
    """
    return torch.mean(a, list(range(1, len(a.shape))))


def vel(m):
    """
    calculates velocity of joints
    :param m: motion (bs X njoints X nfeats X nframes)
    :return: velocity (bs X njoints X nfeats X nframes-1)
    """
    return m[:, :, :, 1:] - m[:, :, :, :-1]