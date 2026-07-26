import os
import numpy as np
import scipy.io as sio
import torch

TREE_SPECIES_DATASET = 'TreeSpeciesHSI'


def _load_first_existing(data_path, candidates, variable_name):
    for candidate in candidates:
        path = os.path.join(data_path, candidate)
        if os.path.exists(path):
            return sio.loadmat(path)[variable_name]
    raise FileNotFoundError('None of these dataset files exist: {}'.format(candidates))


def _resolve_existing_path(data_path, candidates):
    for candidate in candidates:
        path = os.path.join(data_path, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError('None of these dataset files exist: {}'.format(candidates))


def load_mat_array(path, preferred_keys=None):
    """Load a .mat array, supporting both classic and MATLAB v7.3 (HDF5) files."""
    preferred_keys = preferred_keys or []
    try:
        payload = sio.loadmat(path)
        keys = [key for key in payload.keys() if not key.startswith('__')]
        for key in preferred_keys:
            if key in payload:
                return np.asarray(payload[key]), key
        if not keys:
            raise KeyError('No data variables found in {}'.format(path))
        key = keys[0]
        return np.asarray(payload[key]), key
    except NotImplementedError:
        import h5py

        with h5py.File(path, 'r') as handle:
            keys = list(handle.keys())
            key = None
            for candidate in preferred_keys:
                if candidate in handle:
                    key = candidate
                    break
            if key is None:
                if not keys:
                    raise KeyError('No data variables found in {}'.format(path))
                key = keys[0]
            return np.asarray(handle[key]), key


def tree_species_dataset_dir(data_path='./data'):
    return _resolve_existing_path(
        data_path,
        [
            TREE_SPECIES_DATASET,
            'tree-species-hsi-2026',
            '.',
        ],
    )


def _hwc_from_channels_first(array, height, width):
    """Convert (C, d1, d2) cubes into (H, W, C) using target spatial size."""
    if array.ndim != 3:
        raise ValueError('Expected a 3D array, got shape {}'.format(array.shape))

    if array.shape[0] not in (height, width) and array.shape[0] < min(array.shape[1:]):
        # Common MATLAB v7.3 layout: (C, d1, d2)
        c, d1, d2 = array.shape
        if (d1, d2) == (height, width):
            return np.transpose(array, (1, 2, 0))
        if (d1, d2) == (width, height):
            return np.transpose(array, (2, 1, 0))
        raise ValueError(
            'Cannot map channels-first cube {} to spatial size {}x{}'.format(
                array.shape, height, width
            )
        )

    if array.shape == (height, width, array.shape[-1]):
        return array
    if array.shape == (width, height, array.shape[-1]):
        return np.transpose(array, (1, 0, 2))
    raise ValueError(
        'Unsupported HSI layout {} for target spatial size {}x{}'.format(
            array.shape, height, width
        )
    )


def load_tree_species_labels(data_path='./data'):
    dataset_dir = tree_species_dataset_dir(data_path)
    train_label, _ = load_mat_array(
        os.path.join(dataset_dir, 'train_label.mat'),
        preferred_keys=['train_label'],
    )
    val_label, _ = load_mat_array(
        os.path.join(dataset_dir, 'val_label.mat'),
        preferred_keys=['val_label'],
    )
    train_label = np.asarray(train_label).astype(np.int64)
    val_label = np.asarray(val_label).astype(np.int64)
    if train_label.shape != val_label.shape:
        raise ValueError(
            'train_label shape {} does not match val_label shape {}'.format(
                train_label.shape, val_label.shape
            )
        )
    if np.any((train_label > 0) & (val_label > 0)):
        raise ValueError('Official train/val labels overlap; expected disjoint masks.')
    return train_label, val_label


def load_tree_species_train_cube(data_path='./data'):
    dataset_dir = tree_species_dataset_dir(data_path)
    train_label, val_label = load_tree_species_labels(data_path)
    height, width = train_label.shape
    raw, key = load_mat_array(
        os.path.join(dataset_dir, 'data_hsi.mat'),
        preferred_keys=['data', 'image', 'hsi'],
    )
    data = _hwc_from_channels_first(raw, height, width).astype(np.float32)
    gt = np.where(train_label > 0, train_label, val_label).astype(np.int64)
    return data, gt, train_label, val_label, key


def load_tree_species_scene_meta(data_path, scene_name):
    """Load flatten metadata for one competition test scene (no cube I/O)."""
    dataset_dir = tree_species_dataset_dir(data_path)
    scene_info_path = os.path.join(dataset_dir, 'scene_info.csv')
    if not os.path.exists(scene_info_path):
        raise FileNotFoundError(scene_info_path)

    # Normalize names: accept "scene1" or "test_scene1".
    scene_key = scene_name
    if scene_key.startswith('test_'):
        scene_key = scene_key[len('test_'):]

    rows = np.genfromtxt(scene_info_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    if rows.ndim == 0:
        rows = np.array([rows])
    scene_row = None
    for row in rows:
        if str(row['scene']) == scene_key:
            scene_row = row
            break
    if scene_row is None:
        raise KeyError('Scene {} not found in {}'.format(scene_key, scene_info_path))

    height = int(scene_row['height'])
    width = int(scene_row['width'])
    transpose = str(scene_row['transpose']).lower() in ('1', 'true', 'yes')
    flatten_order = str(scene_row['flatten_order']).strip() or 'C'

    candidates = [
        os.path.join(dataset_dir, 'test_{}.mat'.format(scene_key)),
        os.path.join(dataset_dir, '{}.mat'.format(scene_key)),
        os.path.join(dataset_dir, '{}.mat'.format(scene_name)),
    ]
    mat_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            mat_path = candidate
            break
    if mat_path is None:
        raise FileNotFoundError('None of these test cubes exist: {}'.format(candidates))

    return {
        'scene': scene_key,
        'height': height,
        'width': width,
        'transpose': transpose,
        'flatten_order': flatten_order,
        'mat_path': mat_path,
    }


def load_tree_species_test_cube(data_path, scene_name, layout='native_chw'):
    """Load one competition test scene as (H, W, C) float32 plus flatten metadata.

    layout:
      - 'native_chw' (default): match train cube convention. HDF5 (C, d1, d2) -> (d1, d2, C).
        scene_info height/width may be swapped vs (d1, d2); callers must pack submissions
        with pack_tree_species_prediction().
      - 'scene_info_hw': force spatial size (height, width) from scene_info (legacy).
    """
    meta = load_tree_species_scene_meta(data_path, scene_name)
    raw, key = load_mat_array(meta['mat_path'], preferred_keys=['image', 'data', 'hsi'])
    meta = dict(meta)
    meta['variable'] = key
    meta['h5_shape'] = tuple(raw.shape)

    if layout == 'native_chw':
        if raw.ndim != 3:
            raise ValueError('Expected 3D test cube, got {}'.format(raw.shape))
        # Train cubes are stored as (C, d1, d2) with labels shaped (d1, d2).
        if raw.shape[0] <= min(raw.shape[1], raw.shape[2]):
            data = np.transpose(raw, (1, 2, 0)).astype(np.float32)
        elif raw.shape[-1] <= min(raw.shape[0], raw.shape[1]):
            data = np.asarray(raw, dtype=np.float32)
        else:
            data = _hwc_from_channels_first(raw, meta['height'], meta['width']).astype(np.float32)
        meta['layout'] = 'native_chw'
        meta['native_shape'] = data.shape
        return data, meta

    if layout == 'scene_info_hw':
        data = _hwc_from_channels_first(raw, meta['height'], meta['width']).astype(np.float32)
        meta['layout'] = 'scene_info_hw'
        meta['native_shape'] = data.shape
        return data, meta

    raise ValueError('Unsupported test layout: {}'.format(layout))


def pack_tree_species_prediction(pred_hw, meta, transpose=None, flatten_order=None):
    """Pack a spatial prediction map into the competition flat id order.

    pred_hw is the model output on the cube returned by load_tree_species_test_cube.
    We first rewrite it to shape (scene_info.height, scene_info.width), then apply
    scene_info transpose + flatten_order (unless overridden).
    """
    pred_hw = np.asarray(pred_hw)
    height = int(meta['height'])
    width = int(meta['width'])
    if transpose is None:
        transpose = bool(meta['transpose'])
    if flatten_order is None:
        flatten_order = str(meta['flatten_order'])

    if pred_hw.shape == (height, width):
        map_hw = pred_hw
    elif pred_hw.shape == (width, height):
        # native_chw often yields (d1, d2) == (width, height) relative to scene_info.
        map_hw = pred_hw.T
    else:
        raise ValueError(
            'Pred shape {} incompatible with scene_info {}x{}'.format(
                pred_hw.shape, height, width
            )
        )

    arr = map_hw.T if transpose else map_hw
    flat = arr.reshape(-1, order=flatten_order)
    expected = height * width
    if flat.shape[0] != expected:
        raise RuntimeError(
            'Flattened length {} != height*width {}'.format(flat.shape[0], expected)
        )
    return flat.astype(np.int32), map_hw


def load_tree_species_official_split(data_path='./data'):
    train_label, val_label = load_tree_species_labels(data_path)
    train_idx = np.where(train_label.reshape(-1) > 0)[0].astype(np.int64)
    val_idx = np.where(val_label.reshape(-1) > 0)[0].astype(np.int64)
    # Competition has no public test labels; reuse official val for internal test metrics.
    test_idx = val_idx.copy()
    return train_idx, val_idx, test_idx


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

    elif data_set_name == 'QUH-Pingan':
        data = _load_first_existing(
            data_path,
            ['QUH-Pingan.mat', os.path.join('QUH-Pingan', 'QUH-Pingan.mat')],
            'Haigang'
        )
        labels = _load_first_existing(
            data_path,
            [
                'QUH-Pingan_GT.mat',
                'QUH-Pingan_GT(1).mat',
                os.path.join('QUH-Pingan', 'QUH-Pingan_GT.mat'),
                os.path.join('QUH-Pingan', 'QUH-Pingan_GT(1).mat'),
            ],
            'HaigangGT'
        )

    elif data_set_name == 'QUH-Qingyun':
        data = _load_first_existing(
            data_path,
            ['QUH-Qingyun.mat', os.path.join('QUH-Qingyun', 'QUH-Qingyun.mat')],
            'Chengqu'
        )
        labels = _load_first_existing(
            data_path,
            [
                'QUH-Qingyun_GT.mat',
                'QUH-Qingyun_GT(1).mat',
                os.path.join('QUH-Qingyun', 'QUH-Qingyun_GT.mat'),
                os.path.join('QUH-Qingyun', 'QUH-Qingyun_GT(1).mat'),
            ],
            'ChengquGT'
        )

    elif data_set_name == 'QUH-Tangdaowan':
        data = _load_first_existing(
            data_path,
            ['QUH-Tangdaowan.mat', os.path.join('QUH-Tangdaowan', 'QUH-Tangdaowan.mat')],
            'Tangdaowan'
        )
        labels = _load_first_existing(
            data_path,
            ['QUH-Tangdaowan_GT.mat', os.path.join('QUH-Tangdaowan', 'QUH-Tangdaowan_GT.mat')],
            'TangdaowanGT'
        )

    elif data_set_name == TREE_SPECIES_DATASET:
        data, labels, _, _, _ = load_tree_species_train_cube(data_path)

    else:
        raise ValueError('Unsupported dataset: {}'.format(data_set_name))
        
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


def load_fixed_split(split_dir, data_set_name, seed, train_samples, val_samples, gt_reshape=None):
    split_path = os.path.join(
        split_dir,
        data_set_name,
        'seed{}_tr{}_val{}.npz'.format(seed, train_samples, val_samples)
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError('Fixed split file does not exist: {}'.format(split_path))

    payload = np.load(split_path, allow_pickle=False)
    required_keys = ['train_idx', 'val_idx', 'test_idx']
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError('Fixed split file {} is missing keys: {}'.format(split_path, missing_keys))

    train_idx = np.asarray(payload['train_idx'], dtype=np.int64)
    val_idx = np.asarray(payload['val_idx'], dtype=np.int64)
    test_idx = np.asarray(payload['test_idx'], dtype=np.int64)

    if gt_reshape is not None:
        pixel_count = int(gt_reshape.shape[0])
        for name, indices in [('train_idx', train_idx), ('val_idx', val_idx), ('test_idx', test_idx)]:
            if indices.size == 0:
                raise ValueError('{} is empty in fixed split file: {}'.format(name, split_path))
            if int(indices.min()) < 0 or int(indices.max()) >= pixel_count:
                raise ValueError('{} contains out-of-range indices in fixed split file: {}'.format(name, split_path))
            if np.any(gt_reshape[indices] <= 0):
                raise ValueError('{} contains unlabeled pixels in fixed split file: {}'.format(name, split_path))

    return train_idx, val_idx, test_idx, split_path


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
