import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

# root = "E:/UDL/PANCOLLECTION/Data/pansharpening/test_data/GF2/test_gf2_data_FR - 副本.h5"
# file = h5py.File(root, 'r')
# pan = file['pan']
# ms = file['ms']
# lms = file['lms']
# # gt = file['gt']
# length = pan.shape[0]
#
# # 可是将PAN旋转更加简单一些，而且只要保持一致就可以了！
#
#
# pan_list = []
# for index in range(length):
#     rotatedPAN = pan[index, :][0]
#     arr_flipped = np.rot90(np.fliplr(rotatedPAN))  # 可以了，然后把pan都保存替换原来的内容
#     pan_list.append(arr_flipped[np.newaxis, :])
#     plt.imshow(arr_flipped)
#     plt.show()
#
# file_name = "E:/UDL/PANCOLLECTION/Data/pansharpening/test_data/GF2/test_gf2_data_FR.h5"
# f = h5py.File(file_name, 'w')
# f.create_dataset('ms', data=ms)
# # f.create_dataset('gt', data=gt)
# f.create_dataset('pan', data=np.array(pan_list))
# f.create_dataset('lms', data=lms)
# f.close()
""""读取train和val上的结果"""
file_name = "E:/UDL/PANCOLLECTION/Data/pansharpening/test_data/GF2/test_gf2_data_RR - 副本.h5"
f = h5py.File(file_name, 'r')
pan_list = f['pan']
lms_list = f['lms']
ms_list = f['ms']
gt_list = f['gt']
index = 0

pan = pan_list[index,:][0]/2047. #C H W -> H W
gt = np.transpose(gt_list[index,1:4,:,:],(1,2,0))/2047. # n c h w
lms = np.transpose(lms_list[index,1:4,:,:],(1,2,0))/2047.
plt.imshow(pan)
plt.show()
plt.imshow(gt)
plt.show()
plt.imshow(lms)
plt.show()









