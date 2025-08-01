# sys
import h5py
import os
import numpy as np

from sklearn.model_selection import train_test_split
from utils import common

# torch
import torch
from torchvision import datasets, transforms


def load_data_train(_path, _ftype, coords, joints, cycles=3, test_size=0.4):
    print("Loading data...")
    save_path = os.path.join("samples")  # Tạo folder chứa file
    os.makedirs(save_path, exist_ok=True)

    file_feature = os.path.join(_path, 'Data_train' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'Labels_train' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')

    data_list = []
    num_samples = len(ff.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ff_group_key = list(ff.keys())[si]
        data_list.append(list(ff[ff_group_key]))  # Get the data
        time_steps_curr = len(ff[ff_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr

        # Đọc nhãn (dòng cuối cùng trong mảng nhãn)
        raw_label = fl[ff_group_key][()]
        label_index = int(raw_label[-1][0])  # Lấy phần tử cuối, cột đầu
        labels[si] = label_index

                # Lưu mẫu thành file CSV
        # sample_array = np.array(list(ff[ff_group_key]))
        # filename = os.path.join(save_path, f"label_{label_index}_{si}.csv")
        # np.savetxt(filename, sample_array, delimiter=",")
        # print(f"Saved: {filename}")

    data = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
        for ci in range(cycles):
            data[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
    data = common.get_affective_features(np.reshape(data, (data.shape[0], data.shape[1], joints, coords)))[:, :, :48]
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
    return data, labels

def load_data_test(_path, _ftype, coords, joints, cycles=3, test_size=0.4):
    print("Loading data...")
    save_path = os.path.join("samples")  # Tạo folder chứa file
    os.makedirs(save_path, exist_ok=True)

    file_feature = os.path.join(_path, 'Data_test' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'Labels_test' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')

    data_list = []
    num_samples = len(ff.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ff_group_key = list(ff.keys())[si]
        data_list.append(list(ff[ff_group_key]))  # Get the data
        time_steps_curr = len(ff[ff_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr

        # Đọc nhãn (dòng cuối cùng trong mảng nhãn)
        raw_label = fl[ff_group_key][()]
        label_index = int(raw_label[-1][0])  # Lấy phần tử cuối, cột đầu
        labels[si] = label_index

                # Lưu mẫu thành file CSV
        # sample_array = np.array(list(ff[ff_group_key]))
        # filename = os.path.join(save_path, f"label_{label_index}_{si}.csv")
        # np.savetxt(filename, sample_array, delimiter=",")
        # print(f"Saved: {filename}")

    data = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
        for ci in range(cycles):
            data[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
    data = common.get_affective_features(np.reshape(data, (data.shape[0], data.shape[1], joints, coords)))[:, :, :48]
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
    return data, labels

def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, label, joints, coords, num_classes):
        # data: N C T J
        self.data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords, 1))
        self.data = np.moveaxis(self.data, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.N, self.C, self.T, self.J, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label
