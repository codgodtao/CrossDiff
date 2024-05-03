from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py

root = "E:/data/WV3_allmat"
test_data_name = root + "/train"
ms_save_path = test_data_name + '/ms'
pan_save_path = test_data_name + '/pan'

imgs_ms = os.listdir(ms_save_path)  # the name list of the images
imgs_ms.sort(key=lambda x: int(x.split('.')[0]))
imgs_ms = [test_data_name + '/ms/' + img for img in imgs_ms]  # the whole path of images list
imgs_pan = os.listdir(pan_save_path)  # the name list of the images
imgs_pan.sort(key=lambda x: int(x.split('.')[0]))
imgs_pan = [test_data_name + '/pan/' + img for img in imgs_pan]  # the whole path of images list
print('imgs_ms : ', imgs_ms)
print('imgs_pan : ', imgs_pan)
ratio = 4
pan_list = torch.zeros(size=(1, 1, 256, 256))
ms_list = torch.zeros(size=(1, 8, 64, 64))
lms_list = torch.zeros(size=(1, 8, 256, 256))
for item in range(len(imgs_ms)):
# for item in range(10):
    img_path_ms = imgs_ms[item]
    img_path_pan = imgs_pan[item]
    img_ms = scio.loadmat(img_path_ms)['ms'].astype(float)
    img_pan = scio.loadmat(img_path_pan)['pan'].astype(float)

    print(img_ms.shape, img_pan.shape)
    # ref_np = img_ms
    # gai
    ms_size = img_ms.shape[1]
    img_pan = img_pan.reshape(ms_size * 4, ms_size * 4, 1)
    '''data format change'''
    # ref_torch_hwc = torch.from_numpy(ref_np).type(torch.FloatTensor)
    pan_torch_hwc = torch.from_numpy(img_pan).type(torch.FloatTensor)
    ms_torch_hwc = torch.from_numpy(img_ms).type(torch.FloatTensor)

    '''channels change position'''
    pan = pan_torch_hwc.permute(2, 0, 1)  # 1*256*256
    ms = ms_torch_hwc.permute(2, 0, 1)  # 4*64*64

    pan = pan.unsqueeze(0)
    ms = ms.unsqueeze(0)
    print(pan.shape, ms.shape)
    pan_list = torch.concat((pan_list, pan), 0)
    ms_list = torch.concat((ms_list, ms), 0)
# 删除第一个维度的数据,使用基础的切片操作
pan_list = pan_list[1:, :, :, :]
ms_list = ms_list[1:, :, :, :]
lms_list = F.interpolate(ms_list, scale_factor=4, mode="bicubic")  # only 3d,4d,5d support，需要将前面的batchsize设置好
# 在其中第一个维度将他们加起来
print(pan_list.shape, ms_list.shape, lms_list.shape)
#
file_name = "E:/data/full/training_data/train_wv3_data.h5"
f = h5py.File(file_name, 'w')
f.create_dataset('ms', data=ms_list)
f.create_dataset('pan', data=pan_list)
f.create_dataset('lms', data=lms_list)
f.close()

# 训练数据集的格式
# (9000, 4, 64, 64) (9000, 4, 16, 16) (9000, 4, 64, 64) (9000, 1, 64, 64)
# 我们的训练数据集格式
# (6943, 4, 64, 64) (6943, 4, 16, 16) (6943, 4, 64, 64) (6943, 1, 64, 64)
# 验证数据集格式
# (1000, 4, 64, 64) (1000, 4, 16, 16) (1000, 4, 64, 64) (1000, 1, 64, 64)
# 我们的验证数据集格式
# (743, 4, 64, 64) (743, 4, 16, 16) (743, 4, 64, 64) (743, 1, 64, 64)
# 测试数据集格式
# (7, 4, 512, 512) (7, 4, 128, 128) (7, 4, 512, 512) (7, 1, 512, 512)
# 我们的reduced测试数据集格式
# (156, 4, 256, 256) (156, 4, 64, 64) (156, 4, 256, 256) (156, 1, 256, 256)
# full测试数据集格式
# (4,2048,2048) (4,512,512) (1,2048,2048)
# 我们的full测试数据集格式
# (156, 4, 256, 256) (156, 4, 1024, 1024) (156, 1, 1024, 1024)
