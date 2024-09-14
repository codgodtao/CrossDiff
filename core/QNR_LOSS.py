from typing import Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.mylib import get_hp, get_lp, get_hp_gussain


def QIndex_torch(a, b, eps=1e-8):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (torch.Tensor): one-channel images, shape like [N, H, W]
        b (torch.Tensor): one-channel images, shape like [N, H, W]
    Returns:
        torch.Tensor: Q index value of all images
    """
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return torch.mean(4 * cov_ab * E_a * E_b / ((var_a + var_b) * (E_a ** 2 + E_b ** 2) + eps))


def D_lambda_torch(l_ms, ps):
    r"""
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(
                    QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum / L / (L - 1)


def D_s_torch(l_ms, pan, l_pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(
            QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L


def D_s_F_torch(l_ms, pan, l_pan, ps, num):
    "FQNR上使用的空间损失项，用来减少QNR与HQNR存在的弊端"
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    ps = get_hp_gussain(ps, num)
    l_ms = get_hp_gussain(l_ms, num)
    pan = get_hp_gussain(pan, 1)
    l_pan = get_hp_gussain(l_pan, 1)
    for i in range(L):
        sum += torch.abs(
            QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) -
            QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L


# class QNR_loss(nn.Module):
#     def __init__(self):
#         super(QNR_loss, self).__init__()
#
#     def forward(self, pan, ms, out, pan_l=None):
#         # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> tuple[Union[int, Any], Tensor, Tensor]
#         # mean D_lambda value of n images
#         D_lambda = D_lambda_torch(l_ms=ms, ps=out) * 10
#         # mean D_s value of n images
#         D_s = D_s_F_torch(l_ms=ms, pan=pan, l_pan=pan_l if pan_l is not None else down_sample(pan), ps=out, num=4)
#         QNR = (1 - D_lambda) * (1 - D_s)
#         return QNR, D_lambda, D_s


class QNR(nn.Module):
    def __init__(self):
        super(QNR, self).__init__()

    def forward(self, pan, ms, out, pan_l=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> tuple[Union[int, Any], Tensor, Tensor]
        # mean D_lambda value of n images
        D_lambda = D_lambda_torch(l_ms=ms, ps=out)
        # mean D_s value of n images
        D_s = D_s_torch(l_ms=ms, pan=pan, l_pan=pan_l if pan_l is not None else down_sample(pan), ps=out)
        QNR = (1 - D_lambda) * (1 - D_s)
        return QNR, D_lambda, D_s


def down_sample(imgs, r=4, mode='area'):
    r""" down-sample the images

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        r (int): scale ratio, Default: 4
        mode (str): interpolate mode, Default: 'bicubic'
    Returns:
        torch.Tensor: images after down-sampling, shape of [N, C, H//r, W//r]
    """

    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h // r, w // r], mode=mode)


def up_sample(imgs, r=4, mode='area'):
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h * r, w * r], mode=mode)


if __name__ == "__main__":
    loss = QNR()

    pan = torch.randn((1, 1, 256, 256)) * 2047.
    ms = torch.randn((1, 4, 256, 256)) * 2047.
    lms = torch.randn((1, 4, 64, 64)) * 2047.
    sr = torch.randn((1, 4, 256, 256)) * 2047.

    QNR, D_lambda, D_s = loss(pan, lms, sr)
    print(QNR, D_lambda, D_s)
