import sys
import os
# sys.path.append('/content/drive/MyDrive/hyperspectral classification/MambaHSI')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['CUDA_VISIBLE_DEVICES']='0' #选择服务器
import time
import random
import argparse
import numpy as np
import torch
from torchvision import transforms
# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
from model.MambaHSI import ImprovedMambaHSI as MambaHSI
from calflops import calculate_flops
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast, GradScaler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
torch.autograd.set_detect_anomaly(True)

scaler = GradScaler()

accumulation_steps = 4

# 返回当前本地时间 '24-11-28-15.23'
time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path, only_vis_label=True)

# random seed setting 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内建 random 模块的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次计算的结果是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动优化，确保每次运行的一致性


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {'true', '1', 'yes', 'y'}:
        return True
    if lowered in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_model_kwargs(args):
    preset = {}
    if args.model_variant == 'litepyramamba':
        preset = {
            'shared_attention_mode': 'lite_psa',
            'attention_mode': 'none',
            'spatial_mode': 'dwconv_mamba',
            'fusion_mode': 'cross_gate',
            'dynamic_conv_mode': 'shared',
            'cls_hidden_dim': 0,
        }

    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'mamba_type': args.mamba_type,
        'token_num': args.token_num,
        'group_num': args.group_num,
        'attention_mode': args.attention_mode if args.attention_mode is not None else preset.get('attention_mode', 'prca'),
        'shared_attention_mode': args.shared_attention_mode if args.shared_attention_mode is not None else preset.get('shared_attention_mode', 'none'),
        'spatial_mode': args.spatial_mode if args.spatial_mode is not None else preset.get('spatial_mode', 'baseline'),
        'fusion_mode': args.fusion_mode if args.fusion_mode is not None else preset.get('fusion_mode', 'attention'),
        'dynamic_conv_mode': args.dynamic_conv_mode if args.dynamic_conv_mode is not None else preset.get('dynamic_conv_mode', 'dynamic'),
        'cls_hidden_dim': args.cls_hidden_dim if args.cls_hidden_dim is not None else preset.get('cls_hidden_dim', 128),
        'post_mamba_se': args.post_mamba_se,
    }
    return model_kwargs


def build_model(channels, class_count, model_kwargs):
    return MambaHSI(
        in_channels=channels,
        num_classes=class_count,
        **model_kwargs,
    )


def count_trainable_params(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def log_model_statistics(logger, net):
    total_params = count_trainable_params(net)
    logger.info(f'trainable_params={total_params}')

    module_stats = {
        'patch_embedding': count_trainable_params(net.patch_embedding),
        'shared_attention': count_trainable_params(net.shared_attention),
        'mamba_backbone': count_trainable_params(net.mamba),
        'dynamic_conv': count_trainable_params(net.dynamic_conv),
        'cls_head': count_trainable_params(net.cls_head),
    }
    logger.info(f'module_param_breakdown={module_stats}')

    backbone = net.mamba[0]
    if hasattr(backbone, 'spa_mamba') and hasattr(backbone, 'spe_mamba'):
        branch_stats = {
            'spa_mamba': count_trainable_params(backbone.spa_mamba),
            'spe_mamba': count_trainable_params(backbone.spe_mamba),
            'fusion': count_trainable_params(backbone.fusion),
        }
        logger.info(f'branch_param_breakdown={branch_stats}')


def format_model_descriptor(args, net_name, dataset_name, model_kwargs):
    run = sanitize_run_name(args.run_name) or 'default_save_dir'
    return (
        f"model_variant={args.model_variant}, net_name={net_name}, dataset={dataset_name}, run_name={run}, "
        f"mamba_type={model_kwargs['mamba_type']}, hidden_dim={model_kwargs['hidden_dim']}, "
        f"attention_mode={model_kwargs['attention_mode']}, shared_attention_mode={model_kwargs['shared_attention_mode']}, "
        f"spatial_mode={model_kwargs['spatial_mode']}, fusion_mode={model_kwargs['fusion_mode']}, "
        f"dynamic_conv_mode={model_kwargs['dynamic_conv_mode']}, cls_hidden_dim={model_kwargs['cls_hidden_dim']}, "
        f"post_mamba_se={model_kwargs['post_mamba_se']}"
    )


def profile_model_compute(net, input_shape_nchw, logger):
    net.eval()
    params = count_trainable_params(net)
    flops_str = 'N/A'
    macs_str = 'N/A'
    calflops_para = 'N/A'
    err = None
    try:
        flops, macs, calflops_para = calculate_flops(model=net, input_shape=input_shape_nchw)
        flops_str = str(flops)
        macs_str = str(macs)
    except Exception as exc:
        err = str(exc)
        logger.warning(f'calculate_flops failed: {err}')
    return params, flops_str, macs_str, calflops_para, err


def emit_final_model_report(logger, lines):
    banner = '\n==================== Final model report ========================='
    logger.info(banner)
    for line in lines:
        logger.info(line)
        print(line)
    logger.info('================================================================')

def sanitize_run_name(run_name):
    if run_name is None:
        return None
    s = str(run_name).strip()
    if not s:
        return None
    for bad in ('..', os.sep, '/', '\\'):
        s = s.replace(bad, '_')
    return s or None


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=8)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')
    parser.add_argument(
        '--run_name',
        type=str,
        default='',
        help='Optional subfolder under <exp>/<net>/<dataset>/ so parallel ablations do not overwrite each other.',
    )

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=str2bool, default=False)
    parser.add_argument('--model_variant', type=str, default='improved', choices=['improved', 'litepyramamba'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--mamba_type', type=str, default='both', choices=['spa', 'spe', 'both'])
    parser.add_argument('--attention_mode', type=str, default=None, choices=['prca', 'lite_psa', 'none'])
    parser.add_argument('--shared_attention_mode', type=str, default=None, choices=['prca', 'lite_psa', 'none'])
    parser.add_argument('--spatial_mode', type=str, default=None, choices=['baseline', 'dwconv_mamba'])
    parser.add_argument('--fusion_mode', type=str, default=None, choices=['attention', 'cross_gate'])
    parser.add_argument('--dynamic_conv_mode', type=str, default=None, choices=['dynamic', 'shared', 'none'])
    parser.add_argument('--cls_hidden_dim', type=int, default=None)
    parser.add_argument('--post_mamba_se', type=str2bool, default=False)
    parser.add_argument('--token_num', type=int, default=4)
    parser.add_argument('--group_num', type=int, default=4)

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args \
    = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
seed_list = [0, 1, 2]
num_list = [args.train_samples, args.val_samples] # 用于存储训练样本数和验证样本数

dataset_index = args.dataset_index # 选择的数据集索引（如 0、1、2 等），通过索引选择不同的数据集。
max_epoch = args.max_epoch
learning_rate = args.lr
model_kwargs = build_model_kwargs(args)
net_name = 'LitePyraMamba' if args.model_variant == 'litepyramamba' else 'MambaHSI'
paras_dict = {
    'net_name': net_name,
    'dataset_index': dataset_index,
    'num_list': num_list,
    'lr': learning_rate,
    'seed_list': seed_list,
    'model_variant': args.model_variant,
    'run_name': sanitize_run_name(args.run_name) or '',
    **model_kwargs,
}
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','LongKou','Salinas','indian','Botswana','XuZhou','Pavia']
data_set_name = data_set_name_list[dataset_index]
split_image = data_set_name in ['HanChuan', 'Houston','Pavia']

transform = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == '__main__':
    data_set_path = args.data_set_path   # ./data
    work_dir = args.work_dir          #./
    # tr30val10_lr0.0003
    setting_name = 'tr{}val{}'.format(str(args.train_samples), str(args.val_samples)) + '_lr{}'.format(str(learning_rate))
    dataset_name = data_set_name # UP 。。。
    exp_name = args.exp_name

    run_name_safe = sanitize_run_name(args.run_name)
    if run_name_safe:
        save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name, run_name_safe)
    else:
        save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)
    # 路径不存在创建路径
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))
    # './RUNS/MambaHSI/UP/train_tr30_val10.log'

    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache() # 清理 PyTorch 的 GPU 显存缓存
    logger.info(save_folder)

    # Load data and apply preprocessing
    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    # Apply Gaussian filtering
    data_filtered = gaussian_filter(data, sigma=1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=30)
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    data_pca = pca.fit_transform(data_reshaped)
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)

    # Update data shape and other parameters based on PCA-preprocessed data
    height, width, channels = data_pca.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data_pca)
    class_count = int(max(np.unique(gt)))

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.1, 0.01]  # [train_ratio, val_ratio]
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        # 创建实验结果保存目录
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed)) # run0_seed0
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name) # # './RUNS/MambaHSI/UP/run0_seed0'
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis') # './RUNS/MambaHSI/UP/run0_seed0/vis'
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list, num_list, gt_reshape, class_count, flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data_pca, height, width, gt_reshape, index)

        # build Model  单GPU
        net = build_model(channels, class_count, model_kwargs)

        logger.info(paras_dict)
        logger.info(net)
        log_model_statistics(logger, net)

        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(device)
        print(f"x shape: {x.shape}")

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        net.to(device)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

        logger.info(optimizer)
        best_loss = 99999
        if record_computecost:
            net.eval()
            torch.cuda.empty_cache()

            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]))

            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0) # 将 train_label 数据的维度增加一个轴，通常是因为模型期望输入的是一个 4D 张量 (batch_size, channels, height, width)，而 train_label 可能是 3D 的。
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()
            loss_dict = {}

            net.train()

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                # 第一部分前向传播
                y_pred_part1 = net(x_part1)
                ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                # 第二部分前向传播
                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))


            else:
                try:
                    # autocast()：这个上下文管理器启用混合精度训练，它可以减少计算时间并减少显存占用。scaler 用于处理梯度缩放，使得反向传播时能更稳定。
                    with autocast():
                         y_pred = net(x)
                         ls = head_loss(loss_func, y_pred, y_train.long())
                    optimizer.zero_grad()
                    scaler.scale(ls).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()

                # except 处理：如果内存不足或计算出现问题（比如 OOM 错误），会切换为 split_image = True，重新将图像切分为两部分并训练。这是一个容错机制，防止内存不足导致训练崩溃。
                except:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image = True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    y_pred_part1 = net(x_part1)
                    ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    y_pred_part2 = net(x_part2)
                    ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    logger.info('Iter:{}|loss:{}'.format(epoch, (ls1 + ls2).detach().cpu().numpy()))

            torch.cuda.empty_cache()

            # Evaluation stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                output_val = net(x) # 使用验证数据进行前向传播。
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear', align_corners=True)
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy() # 将模型输出的 logits 转换为类别标签
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

                # Save the best model based on validation accuracy
                if OA >= best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)

                if (epoch + 1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)

            torch.cuda.empty_cache()
        toc1 = time.perf_counter()  # 记录结束时间
        train_time = toc1 - tic1  # 计算时间间隔
        logger.info(f"train_time: {train_time} seconds")

        # Final testing phase with the best model
        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        best_net = build_model(channels, class_count, model_kwargs)
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()

        test_evaluator = Evaluator(num_class=class_count)

        with torch.no_grad():
            test_evaluator.reset()
            output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test, size=y_test.shape[1:], mode='bilinear', align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        toc2 = time.perf_counter()  # 记录结束时间
        test_time = toc2 - tic2

        # Save results to file
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)
        Train_Time_ALL.append(train_time)
        Test_Time_ALL.append(test_time)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))

    logger.info('List of OA: {}'.format(list(OA_ALL)))
    logger.info('List of AA: {}'.format(list(AA_ALL)))
    logger.info('List of KPP: {}'.format(list(KPP_ALL)))
    logger.info('OA: {:.2f} ± {:.2f}'.format(np.mean(OA_ALL) * 100, np.std(OA_ALL) * 100))
    logger.info('AA: {:.2f} ± {:.2f}'.format(np.mean(AA_ALL) * 100, np.std(AA_ALL) * 100))
    logger.info('Kpp: {:.2f} ± {:.2f}'.format(np.mean(KPP_ALL) * 100, np.std(KPP_ALL) * 100))
    logger.info('Acc per class: {} ± {}'.format(
        np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2).tolist(),
        np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2).tolist()
    ))
    # 检查训练时间和测试时间数组是否为空
    if len(Train_Time_ALL) > 0:
        avg_train_time = np.mean(Train_Time_ALL)
        std_train_time = np.std(Train_Time_ALL)
    else:
        avg_train_time, std_train_time = 0, 0  # 默认值
        logger.warning("Train_Time_ALL 为空，训练时间无法计算，使用默认值 0。")

    if len(Test_Time_ALL) > 0:
        avg_test_time = np.mean(Test_Time_ALL) * 1000
        std_test_time = np.std(Test_Time_ALL) * 1000
    else:
        avg_test_time, std_test_time = 0, 0  # 默认值
        logger.warning("Test_Time_ALL 为空，测试时间无法计算，使用默认值 0。")

    # 日志记录
    logger.info('Average training time: {:.2f} ± {:.3f}'.format(avg_train_time, std_train_time))
    logger.info('Average testing time: {:.2f} ± {:.3f}'.format(avg_test_time, std_test_time))
    ##############################

    # Save final summary results
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + ' times runs ********************' \
                      + '\nList of OA:' + str(list(OA_ALL)) \
                      + '\nList of AA:' + str(list(AA_ALL)) \
                      + '\nList of KPP:' + str(list(KPP_ALL)) \
                      + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                      + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                      + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                      + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
            np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                      + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
            np.round(np.std(Train_Time_ALL), decimals=3)) \
                      + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
            np.round(np.std(Test_Time_ALL) * 100, decimals=3))
        f.write(str_results)

    model_descriptor = format_model_descriptor(args, net_name, dataset_name, model_kwargs)
    profile_net = build_model(channels, class_count, model_kwargs)
    profile_net.to(device)
    input_shape_nchw = (1, int(channels), int(height), int(width))
    params, flops_str, macs_str, cf_para, flop_err = profile_model_compute(
        profile_net, input_shape_nchw, logger
    )
    final_lines = [
        f"改进/实验配置: {model_descriptor}",
        f"input_shape_NCHW: {input_shape_nchw}",
        f"trainable_params: {params}",
        f"FLOPS (calflops): {flops_str}",
        f"MACs (calflops): {macs_str}",
        f"calflops_reported_params: {cf_para}",
    ]
    if flop_err:
        final_lines.append(f"FLOPS_error: {flop_err}")
    emit_final_model_report(logger, final_lines)
    with open(mean_result_path, 'a', encoding='utf-8') as f:
        f.write("\n\n*************** Final model report (params / FLOPS) ********************\n")
        f.write("\n".join(final_lines) + "\n")

    del profile_net
    del net

# Optional cleanup
torch.cuda.empty_cache()








