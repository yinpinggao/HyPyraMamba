# 精度定量结果与可视化结果 代码整理

---

## 一、精度定量结果

### 1. 评估指标计算核心类 — `utils/evaluation.py`

基于混淆矩阵实现所有定量评估指标。

```python
import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Kappa(self):
        """Cohen's Kappa 系数"""
        xsum = np.sum(self.confusion_matrix, axis=1)
        ysum = np.sum(self.confusion_matrix, axis=0)
        Pe = np.sum(ysum * xsum) * 1.0 / (self.confusion_matrix.sum() ** 2)
        P0 = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        cohens_coefficient = (P0 - Pe) / (1 - Pe)
        return cohens_coefficient

    def ProducerA(self):
        """生产者精度"""
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def UserA(self):
        """用户精度"""
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def Pixel_Accuracy(self):
        """OA: Overall Accuracy，总体精度"""
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        """AA: Average Accuracy，平均类别精度，同时返回每类精度"""
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Mean_Intersection_over_Union(self):
        """mIoU: 平均交并比，同时返回每类 IoU"""
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        """FWIoU: 频率加权交并比"""
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """根据 GT 和预测结果生成混淆矩阵"""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """添加一个 batch 的结果到混淆矩阵"""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
```

**指标总结：**

| 指标 | 方法名 | 含义 |
|------|--------|------|
| OA | `Pixel_Accuracy()` | 总体精度，所有正确分类像素 / 总像素 |
| AA | `Pixel_Accuracy_Class()` | 平均类别精度，各类精度的均值 |
| Kappa | `Kappa()` | Cohen's Kappa 系数，衡量一致性 |
| mIoU | `Mean_Intersection_over_Union()` | 平均交并比 |
| FWIoU | `Frequency_Weighted_Intersection_over_Union()` | 频率加权交并比 |
| PA / UA | `ProducerA()` / `UserA()` | 生产者精度 / 用户精度 |

---

### 2. 辅助函数 — `utils/Loss.py` 中的 `resize`

用于将模型输出 logits 插值到与标签相同的空间尺寸，是计算精度前的必要步骤。

```python
import torch.nn.functional as F

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)
```

---

### 3. 验证阶段精度计算 — `train.py` 第274-289行

每个 epoch 结束后，在验证集上计算 OA、AA、Kappa、mIoU 等指标，并据此保存最优模型。

```python
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
    logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(
        epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

    # 保存验证集上最优模型
    if OA >= best_val_acc:
        best_epoch = epoch + 1
        best_val_acc = OA
        torch.save(net.state_dict(), save_weight_path)
```

---

### 4. 测试阶段精度计算 — `train.py` 第307-335行

加载最优模型后，在测试集上计算最终精度指标。

```python
# Final testing phase with the best model
best_net = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=128)
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
    logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(
        epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test))
```

---

### 5. 单次实验结果保存 — `train.py` 第340-358行

将单次实验的测试精度写入 `result_tr{}_val{}.txt` 文件。

```python
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
f.write(str_results)
f.close()

OA_ALL.append(OA_test)
AA_ALL.append(mAcc_test)
KPP_ALL.append(Kappa_test)
EACH_ACC_ALL.append(Acc_test)
Train_Time_ALL.append(train_time)
Test_Time_ALL.append(test_time)
```

---

### 6. 多次实验汇总统计 — `train.py` 第369-425行

对所有 seed 的实验结果计算 mean±std，并写入 `mean_result.txt`。

```python
OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
EACH_ACC_ALL = np.array(EACH_ACC_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

np.set_printoptions(precision=4)
logger.info('OA: {:.2f} ± {:.2f}'.format(np.mean(OA_ALL) * 100, np.std(OA_ALL) * 100))
logger.info('AA: {:.2f} ± {:.2f}'.format(np.mean(AA_ALL) * 100, np.std(AA_ALL) * 100))
logger.info('Kpp: {:.2f} ± {:.2f}'.format(np.mean(KPP_ALL) * 100, np.std(KPP_ALL) * 100))
logger.info('Acc per class: {} ± {}'.format(
    np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2).tolist(),
    np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2).tolist()
))

# 保存到 mean_result.txt
mean_result_path = os.path.join(save_folder, 'mean_result.txt')
with open(mean_result_path, 'w') as f:
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + ' times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' \
                  + str(np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' \
                  + str(np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' \
                  + str(np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
```

---

## 二、可视化结果

### 1. 可视化核心函数 — `utils/visual_predict.py`

使用 `spectral` 库的 `spy_colors` 调色板，将分类预测图和 GT 保存为彩色 PNG。

```python
import numpy as np
import spectral as spy
from spectral import spy_colors

def visualize_predict(gt, predict_label, save_predict_path, save_gt_path, only_vis_label=False):
    """
    将预测结果可视化为彩色分类图并保存。

    参数:
        gt:                原始 ground truth (H, W)，0 为背景
        predict_label:     预测标签 (1, H, W) 或 (H, W)
        save_predict_path: 预测图保存路径
        save_gt_path:      GT 图保存路径
        only_vis_label:    True 时仅可视化有标签区域（背景置 0），False 时可视化全图
    """
    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label, (row, col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt == 0, gt, predict)
    else:
        vis_predict = predict
    spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    spy.save_rgb(save_gt_path, gt, colors=spy_colors)
```

---

### 2. 可视化封装函数 — `train.py` 第40-42行

每次调用生成两张图：全图预测 + 仅标签区域预测（mask 版本）。

```python
def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    # 全图预测可视化
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    # 仅标签区域可视化 (mask 版本)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path, only_vis_label=True)
```

---

### 3. 训练中可视化（每 50 个 epoch）— `train.py` 第297-300行

在验证阶段，每隔 50 个 epoch 保存一次预测可视化图。

```python
if (epoch + 1) % 50 == 0:
    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)
```

**输出文件：**
- `vis/predict_50.png`, `vis/predict_50_mask.png`
- `vis/predict_100.png`, `vis/predict_100_mask.png`
- `vis/predict_150.png`, `vis/predict_150_mask.png`
- `vis/predict_200.png`, `vis/predict_200_mask.png`
- `vis/gt.png`

---

### 4. 测试阶段最终可视化 — `train.py` 第336行

测试结束后保存最优模型的预测可视化结果。

```python
vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
```

**输出文件：**
- `pred_vis_tr30_val10.png` — 全图预测
- `pred_vis_tr30_val10_mask.png` — 仅标签区域
- `gt_vis_tr30_val10.png` — Ground Truth

---

## 三、输出文件结构总览

```
RUNS/MambaHSI/{数据集名}/
├── train_tr30_val10.log              # 训练日志（含每 epoch 的 OA/AA/Kappa/mIoU）
├── mean_result.txt                   # 多次实验的 mean±std 汇总
├── run0_seed0/
│   ├── best_tr30_val10.pth           # 最优模型权重
│   ├── result_tr30_val10.txt         # 单次实验精度结果
│   ├── pred_vis_tr30_val10.png       # 测试集全图预测可视化
│   ├── pred_vis_tr30_val10_mask.png  # 测试集标签区域预测可视化
│   ├── gt_vis_tr30_val10.png         # Ground Truth 可视化
│   └── vis/
│       ├── gt.png                    # GT 图
│       ├── predict_50.png            # 第 50 epoch 全图预测
│       ├── predict_50_mask.png       # 第 50 epoch 标签区域预测
│       ├── predict_100.png
│       ├── predict_100_mask.png
│       ├── predict_150.png
│       ├── predict_150_mask.png
│       ├── predict_200.png
│       └── predict_200_mask.png
├── run1_seed1/
│   └── ...（同上结构）
└── ...
```

---

## 四、代码执行流程图

```
训练循环 (每个 epoch)
    │
    ├── 模型训练 (前向传播 + 反向传播)
    │
    ├── 验证阶段精度计算
    │   ├── net(x) → resize → argmax → predict
    │   ├── evaluator.add_batch(gt, predict)
    │   ├── 计算 OA / AA / Kappa / mIoU
    │   ├── OA >= best_val_acc → 保存最优模型
    │   └── epoch % 50 == 0 → vis_a_image() 保存可视化
    │
    └── (epoch 循环结束)

测试阶段
    ├── 加载最优模型
    ├── best_net(x) → resize → argmax → predict_test
    ├── test_evaluator.add_batch(gt, predict_test)
    ├── 计算 OA_test / AA_test / Kappa_test / mIoU_test
    ├── vis_a_image() 保存最终可视化
    └── 写入 result_tr{}_val{}.txt

多次实验汇总
    ├── 收集所有 seed 的 OA/AA/Kappa/每类精度
    ├── 计算 mean ± std
    └── 写入 mean_result.txt
```
