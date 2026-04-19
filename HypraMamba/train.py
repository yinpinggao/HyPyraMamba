import os
import time
import random
import argparse
import numpy as np
import torch
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from model.MambaHSI import ImprovedMambaHSI as MambaHSI
from calflops import calculate_flops
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
scaler = GradScaler(enabled=torch.cuda.is_available())
FUSION_NAME = 'ccaf_v2'


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


def compute_balanced_class_weights(train_label, class_count, target_device):
    valid_labels = train_label[train_label >= 0].long().view(-1)
    class_counts = torch.bincount(valid_labels, minlength=class_count).float()
    class_weights = torch.zeros(class_count, dtype=torch.float32)
    nonzero_mask = class_counts > 0

    if nonzero_mask.any():
        total_valid = class_counts[nonzero_mask].sum()
        class_weights[nonzero_mask] = total_valid / (nonzero_mask.sum() * class_counts[nonzero_mask])

    return class_weights.to(target_device), class_counts.long().tolist()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=8)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--class_weight_mode', type=str, default='auto', choices=['auto', 'none', 'balanced'])
    parser.add_argument('--lambda_recon', type=float, default=0.05)
    parser.add_argument('--recon_loss_type', type=str, default='smoothl1')

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args \
    = get_parser()
record_computecost = args.record_computecost
seed_list = [0, 1, 2]
num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index
max_epoch = args.max_epoch
learning_rate = args.lr
lambda_recon = args.lambda_recon
save_net_name = 'MambaHSI_{}'.format(FUSION_NAME)
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','LongKou','Salinas','indian','Botswana','XuZhou','Pavia']
data_set_name = data_set_name_list[dataset_index]
split_image = data_set_name in ['HanChuan', 'Houston','Pavia']

if args.label_smoothing is None:
    label_smoothing = 0.1 if data_set_name == 'indian' else 0.0
else:
    label_smoothing = args.label_smoothing

if args.class_weight_mode == 'auto':
    class_weight_mode = 'balanced' if data_set_name == 'indian' else 'none'
else:
    class_weight_mode = args.class_weight_mode

paras_dict = {
    'net_name': save_net_name,
    'dataset_index': dataset_index,
    'num_list': num_list,
    'lr': learning_rate,
    'seed_list': seed_list,
    'label_smoothing': label_smoothing,
    'fusion_mode': FUSION_NAME,
    'class_weight_mode': class_weight_mode,
    'lambda_recon': lambda_recon,
    'recon_loss_type': args.recon_loss_type,
}

transform = transforms.Compose([
    transforms.ToTensor(),
])


def compute_recon_loss(recon_loss_func, recon_pred, target):
    recon_pred_up = resize(
        input=recon_pred,
        size=target.shape[2:],
        mode='bilinear',
        align_corners=False
    )
    return recon_loss_func(recon_pred_up, target)


def get_fusion_status(model):
    status = 'Fusion mode: {}'.format(FUSION_NAME)
    if model is not None and hasattr(model, 'get_fusion_beta'):
        beta_value = model.get_fusion_beta()
        if beta_value is not None:
            status += '|beta:{:.6f}'.format(beta_value)
    return status

if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    dataset_name = data_set_name

    save_folder = os.path.join(work_dir, args.exp_name, save_net_name, dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))

    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache()
    logger.info(save_folder)
    logger.info(get_fusion_status(model=None))

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    data_filtered = gaussian_filter(data, sigma=1)

    pca = PCA(n_components=30)
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    data_pca = pca.fit_transform(data_reshaped)
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)

    height, width, channels = data_pca.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data_pca)
    class_count = int(max(np.unique(gt)))

    ratio_list = [0.1, 0.01]  # [train_ratio, val_ratio]

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, _ = data_load_operate.sampling(
            ratio_list,
            num_list,
            gt_reshape,
            class_count,
            1,
        )
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(height, width, gt_reshape, index)

        net = MambaHSI(
            in_channels=channels,
            num_classes=class_count,
            hidden_dim=128,
        )

        logger.info(paras_dict)
        logger.info(net)
        logger.info(get_fusion_status(net))

        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(device)

        if class_weight_mode == 'balanced':
            class_weights, class_counts = compute_balanced_class_weights(train_label, class_count, device)
            loss_func = torch.nn.CrossEntropyLoss(
                ignore_index=-1,
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            logger.info('train_class_counts: {}'.format(class_counts))
            logger.info('class_weights: {}'.format([round(v, 4) for v in class_weights.detach().cpu().tolist()]))
        else:
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

        if args.recon_loss_type == 'mse':
            recon_loss_func = torch.nn.MSELoss()
        else:
            recon_loss_func = torch.nn.SmoothL1Loss()

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

        logger.info(optimizer)
        if record_computecost:
            net.eval()
            torch.cuda.empty_cache()

            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]))

            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)

            net.train()

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                # 第一部分前向传播
                y_pred_part1, recon_part1 = net(x_part1, return_aux=True)
                cls_loss_part1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                recon_loss_part1 = compute_recon_loss(recon_loss_func, recon_part1, x_part1)
                ls1 = cls_loss_part1 + lambda_recon * recon_loss_part1
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                # 第二部分前向传播
                y_pred_part2, recon_part2 = net(x_part2, return_aux=True)
                cls_loss_part2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                recon_loss_part2 = compute_recon_loss(recon_loss_func, recon_part2, x_part2)
                ls2 = cls_loss_part2 + lambda_recon * recon_loss_part2
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                logger.info(
                    'Iter:{}|cls_loss:{}|recon_loss:{}|total_loss:{}'.format(
                        epoch,
                        (cls_loss_part1 + cls_loss_part2).detach().cpu().numpy(),
                        (recon_loss_part1 + recon_loss_part2).detach().cpu().numpy(),
                        (ls1 + ls2).detach().cpu().numpy()
                    )
                )


            else:
                try:
                    with autocast(enabled=device.type == 'cuda'):
                        y_pred, recon_pred = net(x, return_aux=True)
                        cls_loss = head_loss(loss_func, y_pred, y_train.long())
                        recon_loss = compute_recon_loss(recon_loss_func, recon_pred, x)
                        ls = cls_loss + lambda_recon * recon_loss
                    optimizer.zero_grad()
                    scaler.scale(ls).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()
                    logger.info(
                        'Iter:{}|cls_loss:{}|recon_loss:{}|total_loss:{}'.format(
                            epoch,
                            cls_loss.detach().cpu().numpy(),
                            recon_loss.detach().cpu().numpy(),
                            ls.detach().cpu().numpy()
                        )
                    )

                except RuntimeError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image = True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    y_pred_part1, recon_part1 = net(x_part1, return_aux=True)
                    cls_loss_part1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                    recon_loss_part1 = compute_recon_loss(recon_loss_func, recon_part1, x_part1)
                    ls1 = cls_loss_part1 + lambda_recon * recon_loss_part1
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    y_pred_part2, recon_part2 = net(x_part2, return_aux=True)
                    cls_loss_part2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                    recon_loss_part2 = compute_recon_loss(recon_loss_func, recon_part2, x_part2)
                    ls2 = cls_loss_part2 + lambda_recon * recon_loss_part2
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    logger.info(
                        'Iter:{}|cls_loss:{}|recon_loss:{}|total_loss:{}'.format(
                            epoch,
                            (cls_loss_part1 + cls_loss_part2).detach().cpu().numpy(),
                            (recon_loss_part1 + recon_loss_part2).detach().cpu().numpy(),
                            (ls1 + ls2).detach().cpu().numpy()
                        )
                    )

            torch.cuda.empty_cache()

            # Evaluation stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                output_val = net(x)
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear', align_corners=True)
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info(get_fusion_status(net))
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

                if OA >= best_val_acc:
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

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()

        load_weight_path = save_weight_path
        best_net = MambaHSI(
            in_channels=channels,
            num_classes=class_count,
            hidden_dim=128,
        )
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        logger.info(get_fusion_status(best_net))

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
            Kappa_test = test_evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        toc2 = time.perf_counter()  # 记录结束时间
        test_time = toc2 - tic2

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
        with open(results_save_path, 'a+') as f:
            f.write(str_results)

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
    if len(Train_Time_ALL) > 0:
        avg_train_time = np.mean(Train_Time_ALL)
        std_train_time = np.std(Train_Time_ALL)
    else:
        avg_train_time, std_train_time = 0, 0
        logger.warning("Train_Time_ALL 为空，训练时间无法计算，使用默认值 0。")

    if len(Test_Time_ALL) > 0:
        avg_test_time = np.mean(Test_Time_ALL) * 1000
        std_test_time = np.std(Test_Time_ALL) * 1000
    else:
        avg_test_time, std_test_time = 0, 0
        logger.warning("Test_Time_ALL 为空，测试时间无法计算，使用默认值 0。")

    logger.info('Average training time: {:.2f} ± {:.3f}'.format(avg_train_time, std_train_time))
    logger.info('Average testing time: {:.2f} ± {:.3f}'.format(avg_test_time, std_test_time))

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

    del net

torch.cuda.empty_cache()
