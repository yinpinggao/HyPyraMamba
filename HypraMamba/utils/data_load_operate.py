import os
import numpy as np
import scipy.io as sio
import torch


def load_data(data_set_name, data_path='./data'):
    if data_set_name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU_gt.mat'))['paviaU_gt']
    elif data_set_name == 'Houston':
        data = sio.loadmat(os.path.join(data_path, 'Houston', 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'Houston', 'Houston_gt.mat'))['houston_gt']
    elif data_set_name == 'HongHu':
        data = sio.loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
    elif data_set_name == 'LongKou':
        data = sio.loadmat(os.path.join(data_path, 'LongKou', 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'LongKou', 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        
    elif data_set_name == 'indian':
        data = sio.loadmat(os.path.join(data_path, 'indian', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian', 'Indian_pines_gt.mat'))['indian_pines_gt']

    elif data_set_name == 'Salinas':
        data = sio.loadmat(os.path.join(data_path, 'Salinas', 'Salinas.mat'))['salinas']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas', 'Salinas_gt.mat'))['salinas_gt']

    elif data_set_name == 'Botswana':
        data = sio.loadmat(os.path.join(data_path, 'Botswana', 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana', 'Botswana_gt.mat'))['Botswana_gt']


    elif data_set_name == 'HanChuan':
        data = sio.loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']

    elif data_set_name == 'XuZhou':
        data = sio.loadmat(os.path.join(data_path, 'XuZhou', 'xuzhou.mat'))['xuzhou']
        labels = sio.loadmat(os.path.join(data_path, 'XuZhou', 'xuzhou_gt.mat'))['xuzhou_gt']

    elif data_set_name == 'Pavia':
        data = sio.loadmat(os.path.join(data_path, 'Pavia', 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia', 'Pavia_gt.mat'))['pavia_gt']
        
    return data, labels


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []

    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class]
            val_index_flag = max(int(ratio_list[1] * len(cls_index)), 1)
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            cls_count = len(cls_index)
            if cls_count >= num_list[0]:
                train_index_flag = num_list[0]
                remaining_after_train = cls_count - train_index_flag
                val_index_flag = min(num_list[1], remaining_after_train)
            else:
                # For rare classes, back off to a 60/20/20 split instead of
                # forcing fixed counts that may exhaust the class.
                train_index_flag = max(int(round(cls_count * 0.6)), 1)
                val_index_flag = max(int(round(cls_count * 0.2)), 0)

                # Keep the split valid after rounding and leave the rest for test.
                if train_index_flag + val_index_flag > cls_count:
                    val_index_flag = max(cls_count - train_index_flag, 0)

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:][val_index_flag:])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:][:val_index_flag])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, val_label_index_list, test_label_index_list, all_label_index_list


def generate_image_iter(hsi_h, hsi_w, label_reshape, index):
    def generate_label_map(num, hsi_w):
        num =np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // hsi_w
        idx_2d[:, 1] = num % hsi_w
        label_map = np.zeros((hsi_h,hsi_w))
        for i in range(num.shape[0]):
            label_map[idx_2d[i, 0], idx_2d[i, 1]] = label_reshape[num[i]]
        return label_map.astype(int)

    # for data label
    train_labels = generate_label_map(index[0], hsi_w) - 1
    val_labels = generate_label_map(index[1], hsi_w) - 1
    test_labels = generate_label_map(index[2], hsi_w) - 1


    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_val = torch.from_numpy(val_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    return y_tensor_train, y_tensor_val, y_tensor_test