import h5py
from torch.utils.data import Dataset
import data.util as Util
import torch
import numpy as np
import skimage.io as io


class PAirMaxDataset(Dataset):
    def __init__(self, dataroot, data_len=-1, phase='train'):
        self.data_len = data_len

        gt = dataroot + "GT.tif"
        pan = dataroot + "PAN.tif"
        lms = dataroot + "MS.tif"
        ms = dataroot + "MS_LR.tif"


        img_scale = 2047.0
        gt1 = np.array(io.imread(gt), dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1).permute(2, 0, 1).unsqueeze(0)

        ms1 = np.array(io.imread(ms), dtype=np.float32) / img_scale
        self.ms = torch.from_numpy(ms1).permute(2, 0, 1).unsqueeze(0)

        lms1 = np.array(io.imread(lms), dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1).permute(2, 0, 1).unsqueeze(0)

        pan1 = np.array(io.imread(pan), dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1).unsqueeze(0).unsqueeze(0)  # Nx1xHxW:
        # logger.info(pan1.shape, lms1.shape, gt1.shape, ms1.shape)
        self.dataset_len = 1  # 目录下所有图像的数量
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        print(self.pan.shape, self.lms.shape, self.gt.shape, self.ms.shape)
        #[b,h,w,c]   (743, c, 64, 64)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = self.gt[index, :, :, :].float()
        img_MS = self.lms[index, :, :, :].float()
        img_PAN = self.pan[index, :, :, :].float()
        img_LR = self.ms[index, :, :, :].float()

        # img_HR = Util.img2res(img_HR, img_MS)  # 残差信息作为输入输出，最终预测结果也是残差信息[-1,1]，需要加上[0,1]的lms
        [img_PAN_norm, img_MS_norm, img_LR_norm] = Util.transform_augment(
            [img_PAN, img_MS, img_LR], min_max=(-1, 1))  # [-1,1]
        # 输入的条件信息为[0,1],预测目标为残差信息[-1,1]
        return {'LR': img_LR_norm, 'PAN': img_PAN_norm, 'MS': img_MS_norm,
                'PAN_0': img_PAN, 'MS_0': img_MS, 'LR_0': img_LR, 'HR': img_HR,
                'Index': index}
