from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py

# # reduced resolution pan256*256 ms64*64*4 lms256*256*4 gt 256*256*4
# # full resolution pan1024*1024 ms256*256*4 lms1024*1024*4 gt不存在
root = "E:/data/WV3_allmat"
test_data_name = root + "/test"
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
ref_list = torch.zeros(size=(1, 8, 256, 256))
pan_list = torch.zeros(size=(1, 1, 256, 256))
ms_list = torch.zeros(size=(1, 8, 64, 64))
lms_list = torch.zeros(size=(1, 8, 256, 256))
for item in range(len(imgs_ms)):
    # for item in range(10):
    img_path_ms = imgs_ms[item]
    img_path_pan = imgs_pan[item]
    img_ms = scio.loadmat(img_path_ms)['ms'].astype(float)
    img_pan = scio.loadmat(img_path_pan)['pan'].astype(float).reshape(1024, 1024, 1)

    ref_np = img_ms
    ms_size = img_ms.shape[1]
    '''To shrink an image, it will generally look best with #INTER_AREA interpolation'''
    pan_np = cv2.resize(img_pan, (ms_size, ms_size), interpolation=cv2.INTER_AREA).reshape(ms_size, ms_size, 1)
    ms_np = cv2.resize(cv2.GaussianBlur(img_ms, (5, 5), 2),
                       (ms_size // ratio, ms_size // ratio), interpolation=cv2.INTER_AREA)  # gai

    pan_torch_hwc = torch.from_numpy(pan_np).type(torch.FloatTensor)
    ms_torch_hwc = torch.from_numpy(ms_np).type(torch.FloatTensor)
    ref_torch_hwc = torch.from_numpy(ref_np).type(torch.FloatTensor)
    '''channels change position'''

    ref = ref_torch_hwc.permute(2, 0, 1)  # 4*64*64
    pan = pan_torch_hwc.permute(2, 0, 1)  # 1*64*64
    ms = ms_torch_hwc.permute(2, 0, 1)  # 4*16*16

    ref = ref.unsqueeze(0)
    pan = pan.unsqueeze(0)
    ms = ms.unsqueeze(0)
    print(pan.shape, ms.shape)
    ref_list = torch.concat((ref_list, ref), 0)
    pan_list = torch.concat((pan_list, pan), 0)
    ms_list = torch.concat((ms_list, ms), 0)
# 删除第一个维度的数据,使用基础的切片操作
ref_list = ref_list[1:, :, :, :]
pan_list = pan_list[1:, :, :, :]
ms_list = ms_list[1:, :, :, :]
lms_list = F.interpolate(ms_list, scale_factor=4, mode="bicubic")  # only 3d,4d,5d support，需要将前面的batchsize设置好
# 在其中第一个维度将他们加起来
print(pan_list.shape, ms_list.shape, lms_list.shape)

# 转换为h5格式
import h5py

file_name = "E:/data/full/training_data/test_wv3_data_RR.h5"
f = h5py.File(file_name, 'w')
f.create_dataset('ms', data=ms_list)
f.create_dataset('gt', data=ref_list)
f.create_dataset('pan', data=pan_list)
f.create_dataset('lms', data=lms_list)
f.close()
