from datetime import datetime
import numpy as np
import argparse
from Trainer import backbone_network
import h5py
import os
import torch
import torch.nn.parallel
import time
import scipy.io
import random
import pandas as pd
from aug_nor import augment_data
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=500, help='Number of total training epochs.')
parser.add_argument('--tr_batch_size', type=int, default=32, help='Size for one training batch.')
parser.add_argument('--te_batch_size', type=int, default=64, help='Size for one testing batch.')
parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
parser.add_argument('--kd_T', default=4.0, type=float, help='T for Temperature scaling')
parser.add_argument('--dim_model', type=int, default=600, help='The length of features in the Mamba')
parser.add_argument('--dim_channel', type=int, default=6, help='The length of features in the SSD')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--dropout', default=0.05, type=float, help='Dropout')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--cls_lambda', type=float, default=1.5, help='分类损失权重')
parser.add_argument('--t1_lambda', type=float, default=1, help='loss_t1 权重')
parser.add_argument('--t2_lambda', type=float, default=1, help='loss_t2 权重')
parser.add_argument('--re_lambda', type=float, default=1, help='loss_re 权重')
parser.add_argument('--dist_lambda', type=float, default=1, help='loss_dist1 权重')

# 解析参数
args = parser.parse_args()
opt = vars(args)

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(0)

init_time = time.time()

##### 数据加载
#####Dataloader
#------------------------- 读取数据 -------------------------
savepath = 'D:\dataset\demo\\filter_DE\\1280_32_60_6\\final_combined.mat'

X = scipy.io.loadmat(savepath)['data'] # (1280,32,60,6)

Y = scipy.io.loadmat(savepath)['labels']  # (32,40,4)

X=X[:,:,:,1]############选择不同波段
Ytemp = Y[:, 0]

X_torch = torch.from_numpy(X).float()
Y_torch = torch.from_numpy(Ytemp).long()
Y_torch = (Y_torch >= 5).long()



#####增大样本
data, labels = augment_data(X_torch, Y_torch, times=2)

##### 创建 CSV 文件保存结果（写入表头）
# 新增保存的超参数：num_epoch, dropout, cls_lambda, t1_lambda, t2_lambda, re_lambda, dist_lambda
columns = ['time', 'Fold', 'beEpoch', 'epo_beAcc', 'epo_beSen', 'epo_beSpe', 'epo_beF1', 'epo_bePre', 'epo_beRec',
           'num_epoch', 'dropout', 'cls_lambda', 't1_lambda', 't2_lambda', 're_lambda', 'dist_lambda']
df = pd.DataFrame(columns=columns)
root_path = 'D:\\Users\\Administrator\\PycharmProjects\\LM-KD-main\\LKD1/results'
os.makedirs(root_path, exist_ok=True)  # 如果目录不存在则创建
local_time = time.localtime()[0:5]
csv_name = 'Cross_{:02d}_{:02d}{:02d}_{:02d}{:02d}.csv'.format(local_time[0], local_time[1], local_time[2],
                                                                local_time[3], local_time[4])
csv_path = os.path.join(root_path, csv_name)
df.to_csv(csv_path, index=False)

global_start_time = time.time()

# 定义 KFold 参数
K = 5  # 可根据需要调整为 5、10等
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# 用于保存每一折的最佳指标
fold_results = []

# 开始 k 折交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    # 分离训练集和测试集数据
    data_train, data_test = data[train_idx], data[test_idx]
    lab_train, lab_test = labels[train_idx], labels[test_idx]

    # 设置 t 的值（确保 300 能被 t 整除）
    t = 3
    if 300 % t != 0:
        raise ValueError("300 必须能被 t 整除，以便进行正确的划分。")

    # 更新 batch_size 参数
    opt['te_batch_size'] = lab_test.shape[0]
    opt['tr_batch_size'] = lab_train.shape[0] // 3

    print("Fold {}: Train shape: {}".format(fold + 1, (data_train.shape[0], data_train.shape[1], data_train.shape[2])))
    print("Fold {}: Test shape: {}".format(fold + 1, (data_test.shape[0], data_test.shape[1], data_test.shape[2])))
    print("Train_batch_num:", lab_train.shape[0] // opt['tr_batch_size'])
    print("Test_batch_num:", lab_test.shape[0] // opt['te_batch_size'])

    ## 构建模型
    model = backbone_network(opt)
    model.cuda()

    correct = 0  # 记录该折中最佳测试准确率
    be_acc0 = 0
    be_acc1 = 0
    be_sen = 0
    be_spe = 0
    be_f1 = 0
    be_pre = 0
    be_rec = 0
    best_epo = 0

    # 创建训练数据集和 DataLoader
    train_dataset = TensorDataset(data_train, lab_train)
    train_loader = DataLoader(train_dataset, batch_size=opt['tr_batch_size'], shuffle=True)

    # 训练若干个 epoch
    for epoch in range(1, args.num_epoch + 1):
        train_loss = 0
        train_acc = 0
        train_acc0 = 0
        train_acc1 = 0
        train_sen = 0
        train_spe = 0
        train_f1 = 0
        train_pre = 0
        train_rec = 0
        count = 0

        # 训练循环
        for tr_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.float().cuda()
            train_y = train_y.long().cuda()

            # 模型前向传播及反向传播
            log, loss = model.train(train_x, train_y)

            # 计算预测类别
            _, pred_class = torch.max(log.cpu(), dim=1)
            train_y = train_y.cpu()
            unique_labels = np.unique(train_y)
            if len(unique_labels) < 2:
                print(f"Train_Batch {tr_idx + 1} only contains one class. Skipping metrics calculation.")
                continue

            accuracy = accuracy_score(train_y, pred_class)
            precision = precision_score(train_y, pred_class, average='weighted', zero_division=0)
            recall = recall_score(train_y, pred_class, average='weighted', zero_division=0)
            f1 = f1_score(train_y, pred_class, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(train_y, pred_class)
            tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            count += 1

            # 计算每个标签的准确率
            label_accuracies = {}
            for label in unique_labels:
                label_indices = train_y == label
                label_accuracy = accuracy_score(train_y[label_indices], pred_class[label_indices])
                label_accuracies[label] = label_accuracy

            train_loss += loss
            train_acc += accuracy
            train_acc0 += label_accuracies.get(0, 0)
            train_acc1 += label_accuracies.get(1, 0)
            train_sen += sensitivity
            train_spe += specificity
            train_f1 += f1
            train_pre += precision
            train_rec += recall

        # 计算所有 batch 的平均值
        train_acc /= count
        train_acc0 /= count
        train_acc1 /= count
        train_sen /= count
        train_spe /= count
        train_f1 /= count
        train_pre /= count
        train_rec /= count
        train_loss /= count

        print(f"Fold = {fold + 1}: Epoch = {epoch}: "
              f"Train Accuracy = {train_acc:.4f}, Sensitivity = {train_sen:.4f}, Specificity = {train_spe:.4f}, "
              f"F1-Score = {train_f1:.4f}, Precision = {train_pre:.4f}, Recall = {train_rec:.4f}, "
              f"Train Loss = {train_loss:.4f}")

        # 评估模型在测试集上的表现
        test_loss = 0
        test_acc = 0
        test_acc0 = 0
        test_acc1 = 0
        test_sen = 0
        test_spe = 0
        test_f1 = 0
        test_pre = 0
        test_rec = 0
        count = 0

        test_dataset = TensorDataset(data_test, lab_test)
        test_loader = DataLoader(test_dataset, batch_size=opt['te_batch_size'], shuffle=True)

        for te_idx, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.float().cuda()
            test_y = test_y.long().cuda()

            predicts = model.predict(test_x, test_y)
            _, pred_class = torch.max(predicts.cpu(), dim=1)
            test_y = test_y.cpu()
            unique_labels = np.unique(test_y)
            if len(unique_labels) < 2:
                print(f"Test_Batch {te_idx + 1} only contains one class. Skipping metrics calculation.")
                continue

            accuracy = accuracy_score(test_y, pred_class)
            precision = precision_score(test_y, pred_class, average='weighted', zero_division=0)
            recall = recall_score(test_y, pred_class, average='weighted', zero_division=0)
            f1 = f1_score(test_y, pred_class, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(test_y, pred_class)
            tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            count += 1

            label_accuracies = {}
            for label in unique_labels:
                label_indices = test_y == label
                label_accuracy = accuracy_score(test_y[label_indices], pred_class[label_indices])
                label_accuracies[label] = label_accuracy

            test_acc += accuracy
            test_acc0 += label_accuracies.get(0, 0)
            test_acc1 += label_accuracies.get(1, 0)
            test_sen += sensitivity
            test_spe += specificity
            test_f1 += f1
            test_pre += precision
            test_rec += recall

        test_acc /= count
        test_acc0 /= count
        test_acc1 /= count
        test_sen /= count
        test_spe /= count
        test_f1 /= count
        test_pre /= count
        test_rec /= count

        print(f"Fold = {fold + 1}: Epoch = {epoch}: "
              f"Test Accuracy = {test_acc:.4f}, Sensitivity = {test_sen:.4f}, Specificity = {test_spe:.4f}, "
              f"F1-Score = {test_f1:.4f}, Precision = {test_pre:.4f}, Recall = {test_rec:.4f}\n")

        # 记录最佳指标（取测试准确率最高的 epoch）
        if test_acc > correct:
            correct = test_acc
            be_acc0 = test_acc0
            be_acc1 = test_acc1
            be_sen = test_sen
            be_spe = test_spe
            be_f1 = test_f1
            be_pre = test_pre
            be_rec = test_rec
            best_epo = epoch

    # 输出当前折的最佳结果
    print(f"Fold = {fold + 1}: Best Epoch = {best_epo}: Best_Accuracy = {correct:.4f}, "
          f"Best_Sensitivity = {be_sen:.4f}, Best_Specificity = {be_spe:.4f}, "
          f"Best_F1 = {be_f1:.4f}, Best_Precision = {be_pre:.4f}, Best_Recall = {be_rec:.4f}\n")

    # 将当前折的结果保存到列表中
    fold_results.append([fold + 1, best_epo, correct, be_sen, be_spe, be_f1, be_pre, be_rec])

    # 保存到 CSV（追加方式），增加超参数信息
    timestr = datetime.now().strftime("%m%d%H%M%S")
    row = [timestr, fold + 1, best_epo, correct, be_sen, be_spe, be_f1, be_pre, be_rec,
           args.num_epoch, args.dropout, args.cls_lambda, args.t1_lambda, args.t2_lambda, args.re_lambda, args.dist_lambda]
    res_df = pd.DataFrame([row], columns=columns)
    res_df.to_csv(csv_path, mode='a', header=False, index=False)

# 所有折结束后，转换结果为 DataFrame，并计算平均指标
fold_results_df = pd.DataFrame(fold_results,
                               columns=['Fold', 'beEpoch', 'epo_beAcc', 'epo_beSen', 'epo_beSpe', 'epo_beF1',
                                        'epo_bePre', 'epo_beRec'])
print("每一折的结果：")
print(fold_results_df)

avg_results = fold_results_df.mean(numeric_only=True)
print("\n各指标在所有折上的平均值：")
print(avg_results)

# 将平均指标也保存到 CSV 中，标记为 'Average'
avg_row = ['Average', 'Average', avg_results['beEpoch'], avg_results['epo_beAcc'], avg_results['epo_beSen'],
           avg_results['epo_beSpe'], avg_results['epo_beF1'], avg_results['epo_bePre'], avg_results['epo_beRec'],
           args.num_epoch, args.dropout, args.cls_lambda, args.t1_lambda, args.t2_lambda, args.re_lambda, args.dist_lambda]
avg_df = pd.DataFrame([avg_row], columns=columns)
avg_df.to_csv(csv_path, mode='a', header=False, index=False)

duration = time.time() - global_start_time
print("Duration time:", duration)
