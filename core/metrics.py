import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch


# def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
#     # 再次进行一轮标准化，归一为[0-1]分布
#     tensor = (tensor - min_max[0]) / \
#              (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(
#             math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type)


def tensor2img_4C(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    修改输入通道数量，由RGB到包含4个或8个通道的图像，但是只取其中可见光的三个通道[4,256,256]
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2 or n_dim == 1:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        # 保存需要的sr结果

    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    # if mode == 'gray':
    #     cv2.imwrite(img_path, img)
    # else:
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def postprocess(images):
    return [tensor2img_4C(image) for image in images]


def set_seed(seed, gl_seed=0):
    """  set random seed, gl_seed used in worker_init_fn function """
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
        speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
    if seed >= 0 and gl_seed >= 0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_gpu(args, distributed=False, rank=0):
    """ set parameter to gpu or ddp """
    if args is None:
        return None
    if distributed and isinstance(args, torch.nn.Module):
        return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True,
                   find_unused_parameters=True)
    else:
        return args.cuda()


def set_device(args, distributed=False, rank=0):
    """ set parameter to gpu or cpu """
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (set_gpu(item, distributed, rank) for item in args)
        elif isinstance(args, dict):
            return {key: set_gpu(args[key], distributed, rank) for key in args}
        else:
            args = set_gpu(args, distributed, rank)
    return args


# def load_image(path):
#     """ Load .TIF image to np.array
#
#     Args:
#         path (str): path of TIF image
#     Returns:
#         np.array: value matrix in [C, H, W] or [H, W]
#     """
#     img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
#     return img
#
#
# def save_image(path, array):
#     """ Save np.array as .TIF image
#
#     Args:
#         path (str): path to save as TIF image
#         np.array: shape like [C, H, W] or [H, W]
#     """
#     # Meaningless Default Value
#     raster_origin = (-123.25745, 45.43013)
#     pixel_width = 2.4
#     pixel_height = 2.4
#
#     if array.ndim == 3:
#         chans = array.shape[0]
#         cols = array.shape[2]
#         rows = array.shape[1]
#         origin_x = raster_origin[0]
#         origin_y = raster_origin[1]
#
#         driver = gdal.GetDriverByName('GTiff')
#
#         out_raster = driver.Create(path, cols, rows, chans, gdal.GDT_UInt16)
#         # print(path, cols, rows, chans, out_raster)
#         out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))
#         for i in range(1, chans + 1):
#             out_band = out_raster.GetRasterBand(i)
#             out_band.WriteArray(array[i - 1, :, :])
#         out_raster_srs = osr.SpatialReference()
#         out_raster_srs.ImportFromEPSG(4326)
#         out_raster.SetProjection(out_raster_srs.ExportToWkt())
#         out_band.FlushCache()
#     elif array.ndim == 2:
#         cols = array.shape[1]
#         rows = array.shape[0]
#         origin_x = raster_origin[0]
#         origin_y = raster_origin[1]
#
#         driver = gdal.GetDriverByName('GTiff')
#
#         out_raster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
#         out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))
#
#         out_band = out_raster.GetRasterBand(1)
#         out_band.WriteArray(array[:, :])


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
