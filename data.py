import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torch.autograd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import os
import torch

batch_size = 128 #一批128个
num_epoch = 100 #总共100批
z_dimension = 200 #噪音维度
input_dimension = 1024 #输入维度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

spoof_data3 = np.load("/new_data/yhang/GNSS/Complex/complex_ds3/complex_spoof_ds3.npy")
spoof_data4 = np.load("/new_data/yhang/GNSS/Complex/complex_ds4/complex_spoof_ds4.npy")
# spoof_data8 = np.load("/new_data/yhang/GNSS/Complex/complex_ds8/complex_spoof_ds8.npy")
clean_data = np.load("/new_data/yhang/GNSS/Complex/complex_clean/clean.npy")

signal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5),std=(0.5))
])

# 创建标签，对于'real'类别，我们使用0
clean_labels = np.zeros((clean_data.shape[0], 100, 1,))
# 对于'spoof_data3'类别，我们使用1
spoof_data3_labels = np.ones((spoof_data3.shape[0], 100, 1,))
# 对于'spoof_data4'类别，我们使用2
spoof_data4_labels = np.ones((spoof_data4.shape[0], 100, 1,)) * 2
# 对于'spoof_data8'类别，我们使用3
# spoof_data8_labels = np.ones((spoof_data8.shape[0],)) * 3

# 同样，将所有标签在第一维上合并
labels = np.vstack((clean_labels, spoof_data3_labels, spoof_data4_labels))
# 对标签进行维度修改改成(labels.shape[0], 1)
print(labels.shape)

# 将所有数据在第一维上合并
data = np.vstack((clean_data, spoof_data3, spoof_data4))

# 拆分数据为训练集和测试集
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)
np.save("/new_data/yhang/GNSS/TEXBAT/train/ds34/data.npy", data_train)
np.save("/new_data/yhang/GNSS/TEXBAT/train/ds34/label.npy", labels_train)
np.save("/new_data/yhang/GNSS/TEXBAT/test/ds34/data.npy", data_test)
np.save("/new_data/yhang/GNSS/TEXBAT/test/ds34/label.npy", labels_test)

print(data_train.shape)
print(labels_train.shape)
print(data_test.shape)
print(labels_test.shape)