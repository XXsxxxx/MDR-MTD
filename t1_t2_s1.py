import math

from torch import nn
from click.core import batch
from torch import nn
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import random
from sklearn import metrics
import torch
from torch_geometric.nn import ChebConv, global_mean_pool as gap
from transformers import GPT2Model, GPT2Config
import os
from scipy.io import loadmat
import numpy as np
from ms_ast_gcn import grad_reverse

CUDA_LAUNCH_BLOCKING=1
#------------------------- 教师1模型 -------------------------
class CrossTransformer(nn.Module):
    def __init__(self, d1, d2, seq_length, feature_dim, outputdim):
        super(CrossTransformer, self).__init__()
        self.fc_map_a = nn.Linear(d1, feature_dim)
        self.fc_map_b = nn.Linear(d2, feature_dim)
        self.layer_norm_a = nn.LayerNorm(normalized_shape=[seq_length, d1])
        self.layer_norm_b = nn.LayerNorm(normalized_shape=[seq_length, d2])
        self.dropout = nn.Dropout(0.1)

        self.feature_dim = feature_dim
        self.seq_length = seq_length

        # 位置编码
        self.pos_encoder = nn.Embedding(seq_length, feature_dim)

        # 自注意力
        self.self_attn1 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.self_attn2 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)

        # 交叉注意力
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)

        # 融合投影到 gpt_n_embd
        self.fusion_linear = nn.Linear(feature_dim * 2, outputdim)

    def forward(self, seq1, seq2):
        seq1 = self.layer_norm_a(seq1)
        seq1 = self.dropout(F.relu(self.fc_map_a(seq1)))

        seq2 = self.layer_norm_b(seq2)
        seq2 = self.dropout(F.relu(self.fc_map_b(seq2)))

        # 位置编码
        pos = torch.arange(self.seq_length, device=seq1.device).unsqueeze(0).repeat(seq1.size(0), 1)
        seq1 = seq1 + self.pos_encoder(pos)
        seq2 = seq2 + self.pos_encoder(pos)

        # 自注意力
        seq1 = self.self_attn1(seq1.permute(1, 0, 2)).permute(1, 0, 2)
        seq2 = self.self_attn2(seq2.permute(1, 0, 2)).permute(1, 0, 2)

        # 交叉注意力
        seq1_att, _ = self.cross_attn1(seq1, seq2, seq2)
        seq2_att, _ = self.cross_attn2(seq2, seq1, seq1)

        # 合并特征
        combined_features = torch.cat([seq1_att, seq2_att], dim=-1)
        combined_features = torch.mean(combined_features, dim=1)  # 融合时序信息
        fused_features = self.fusion_linear(combined_features)    # 投影到 gpt_n_embd
        return fused_features

#------------------------- 教师1模型 -------------------------
class DGCNN(nn.Module):
    def __init__(self, num_electrodes=32, in_channels=60, num_embed=2, k=2, relu_is=1, layers=None, dropout_rate=0.5):
        super(DGCNN, self).__init__()
        if layers is None:
            layers = [128]
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_embed
        self.relu_is = relu_is

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(GraphConv(self.k, self.in_channels, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.graphConvs.append(GraphConv(self.k, self.layers[i], self.layers[i + 1]))

        self.fc = nn.Linear(self.num_electrodes * self.layers[-1], num_embed, bias=True)
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList()
        for i in range(len(self.layers)):
            self.b_relus.append(B1ReLU(self.layers[i]))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class B1ReLU(nn.Module):
    def __init__(self, bias_shape):
        super(B1ReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)

def laplacian(w):
    d = torch.sum(w, dim=1)
    d_re = 1 / torch.sqrt(d + 1e-5)
    d_matrix = torch.diag_embed(d_re)
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap

class GraphConv(nn.Module):
    def __init__(self, k, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        t = torch.ones(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
        if self.k == 1:
            return t.unsqueeze(1)
        if self.k == 2:
            return torch.cat((t.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        elif self.k > 2:
            tk_minus_one = x
            tk = torch.matmul(lap, x)
            t = torch.cat((t.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)
            for i in range(3, self.k):
                tk_minus_two, tk_minus_one = tk_minus_one, tk
                tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
                t = torch.cat((t, tk.unsqueeze(1)), dim=1)
            return t

    def forward(self, x, lap):
        cp = self.chebyshev_polynomial(x, lap)   # (batch, k, ele_channel, in_channel)
        cp = cp.permute(0, 2, 3, 1)             # (batch, ele_channel, in_channel, k)
        cp = cp.flatten(start_dim=2)            # (batch, ele_channel, in_channel*k)
        out = torch.matmul(cp, self.weight)
        return out
#------------------------- 教师2模型 -------------------------

# 用于去除卷积因 padding 导致的多余时间步，确保因果卷积
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 去除最后 chomp_size 个时间步
        return x[:, :, :-self.chomp_size].contiguous()


# TCN 的残差块，包含两层膨胀卷积、ReLU、Dropout 以及残差连接
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        # 若输入输出通道数不同，则用 1x1 卷积进行匹配
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN 模块，由多个 TemporalBlock 依次叠加构成
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: 输入通道数，例如 EEG 信号的通道数 32
            num_channels: 每层输出通道数列表，例如 [64, 128]
            kernel_size: 卷积核大小
            dropout: dropout 概率
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀因子按层递增
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 为保持输出长度一致，padding = (kernel_size - 1) * dilation_size
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 输入 x: (batch_size, num_inputs, sequence_length)
        return self.network(x)


# -------------------- TCN 分类器 --------------------
class TCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: 输入通道数（例如 EEG 信号的通道数 32）
            num_channels: TCN 中各层的输出通道数列表，如 [64, 128]
            num_classes: 分类数（例如 2）
            kernel_size: 卷积核大小
            dropout: dropout 比例
        """
        super(TCNClassifier, self).__init__()
        self.tcn = TCN(num_inputs, num_channels, kernel_size, dropout)
        # 全局平均池化，用于将时间维度压缩为 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x 形状: (batch_size, num_inputs, sequence_length)
        out = self.tcn(x)  # 输出形状: (batch_size, num_channels[-1], sequence_length)
        out = self.global_pool(out)  # 输出形状: (batch_size, num_channels[-1], 1)
        out = out.squeeze(-1)  # 压缩最后一个维度 -> (batch_size, num_channels[-1])
        logits = self.fc(out)  # 最终 logits 形状: (batch_size, num_classes)
        return logits

#------------------------- 学生模型 -------------------------
class EEGNet(nn.Module):
    """
    轻量化学生网络，基于 EEGNet 架构设计
    输入数据形状: (batch_size, num_channels, num_samples)
    输出 logits: (batch_size, num_classes)
    """

    def __init__(self, num_channels=32, num_samples=60, num_classes=2,
                 dropoutRate=0.5, F1=8, D=2, F2=16, kernel_length=8):
        """
        Args:
            num_channels: 输入 EEG 信号的通道数（如 32）
            num_samples: 时间步数（如 60）
            num_classes: 分类数（如 2）
            dropoutRate: dropout 概率
            F1: 第一层卷积的滤波器个数
            D: 深度卷积扩展因子，决定深度卷积后滤波器数 = F1 * D
            F2: 分离卷积输出的滤波器个数
            kernel_length: 第一层时域卷积的核尺寸
        """
        super(EEGNet, self).__init__()
        # 第一层：时域卷积（1×kernel_length），对时间维度进行滤波
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        # 第二层：深度卷积，作用在空间维度上（跨通道），卷积核尺寸为 (num_channels, 1)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        # 第三层：分离卷积（先深度卷积后逐点卷积）进一步提取特征
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 8), padding=(0, 8 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropoutRate)
        )
        # 计算经过两次池化后时间维度的长度：
        # 初始时间步数: num_samples -> 第一次池化：num_samples/2 -> 第二次池化：num_samples/4
        out_time = num_samples // 4
        # 全连接层：将卷积提取的特征展开后映射到分类数
        self.classifier = nn.Linear(F2 * out_time, num_classes)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 (batch_size, num_channels, num_samples)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 为适应 2D 卷积，将输入扩展为 (batch_size, 1, num_channels, num_samples)
        x = x.unsqueeze(1)
        out = self.firstconv(x)  # 输出形状: (batch_size, F1, num_channels, num_samples)
        out = self.depthwiseConv(out)  # 输出形状: (batch_size, F1*D, 1, num_samples/2)
        out = self.separableConv(out)  # 输出形状: (batch_size, F2, 1, num_samples/4)
        out = out.squeeze(2)  # 压缩掉通道维度，将形状变为 (batch_size, F2, num_samples/4)
        out = out.flatten(1)  # 展平成 (batch_size, F2 * (num_samples/4))
        logits = self.classifier(out)  # 输出 logits: (batch_size, num_classes)
        return logits


#------------------------- 其他函数 -------------------------
class CAMKD(nn.Module):
    def __init__(self):
        super(CAMKD, self).__init__()
        self.crit_ce = nn.CrossEntropyLoss(reduction='none')
        self.crit_mse = nn.MSELoss(reduction='none')

    def forward(self, trans_feat_s_list, mid_feat_t_list, output_feat_t_list, target):
        bsz = target.shape[0]
        loss_t = [self.crit_ce(logit_t, target) for logit_t in output_feat_t_list]
        num_teacher = len(trans_feat_s_list)
        loss_t = torch.stack(loss_t, dim=0)
        weight = (1.0 - F.softmax(loss_t, dim=0)) / (num_teacher - 1)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
            tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t).reshape(bsz, -1).mean(-1)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0)
        loss = torch.mul(weight, loss_st).sum()
        loss /= (1.0 * bsz * num_teacher)
        return loss, weight

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if is_ca:
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss

def init_weights(m, feature_dim):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=feature_dim ** -0.5)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)

def find_max_values_and_indices(data_list):
    if not data_list:
        return None
    max_value = max(data_list)
    indices = [index for index, value in enumerate(data_list) if value == max_value]
    return max_value, indices

def find_max_value_by_indices(array, indices):
    if not array or not indices:
        return None
    max_value = array[indices[0]]
    max_index = indices[0]
    for index in indices[1:]:
        if array[index] > max_value:
            max_value = array[index]
            max_index = index
    return max_index, max_value

def init_edge_weight():
    adj_localglobal = loadmat('../LKD_HCI/adj_mat32_saved.mat')
    adj_localglobal = adj_localglobal['adj_mat']

    adj_predefined = np.ones(shape=(adj_localglobal.shape[0], adj_localglobal.shape[1]))

    index = np.argwhere(adj_localglobal == 0)
    adj_predefined[index[:, 0], index[:, 1]] = 0.5
    adj_predefined[range(adj_predefined.shape[0]), range(adj_predefined.shape[0])] = 0

    adj_t = torch.tensor(adj_predefined)
    edge_index = np.array(adj_t.nonzero().t().contiguous())

    initialized_edge_weight = adj_predefined[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)

    return initialized_edge_weight#



class adj_update(nn.Module):
    def __init__(self, inc, reduction_ratio):
        super(adj_update, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias = False),
                                nn.ELU(inplace = False),
                                nn.Linear(inc // reduction_ratio, inc, bias = False),
                                nn.Tanh(),
                                nn.ReLU(inplace = False))

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x



class Encoder(torch.nn.Module):
    def __init__(self, num_features, channels=32):
        super(Encoder, self).__init__()

        self.head = nn.Identity()
        self.conv1 = ChebConv(num_features, 256, K=5)
        self.conv2 = ChebConv(256, 64, K=5)

        num_edges = channels * channels - channels
        self.edge_weight = nn.Parameter(torch.tensor(init_edge_weight(), dtype = torch.float32, requires_grad = True))
        self.adj_update = adj_update(num_edges, reduction_ratio=4)

    def forward(self, x, edge_index):
        edge_weight = self.edge_weight
        train_edge_weight = self.adj_update(edge_weight)
        _edge_weight = train_edge_weight
        for i in range(edge_index.shape[-1] // train_edge_weight.shape[0] - 1):
            train_edge_weight = torch.cat((train_edge_weight, _edge_weight), dim=0)

        x = F.relu(self.conv1(x, edge_index, train_edge_weight))
        x = self.conv2(x, edge_index, train_edge_weight)
        x = self.head(x)

        return x, _edge_weight, train_edge_weight



class Encoder_to_Decoder(torch.nn.Module):
    def __init__(self):
        super(Encoder_to_Decoder, self).__init__()
        self.encoder_to_decoder = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        x = self.encoder_to_decoder(x)
        return x



class Decoder(torch.nn.Module):
    def __init__(self, num_features):
        super(Decoder, self).__init__()

        self.head = nn.Identity()
        self.conv1 = ChebConv(64, 256, K=5)
        self.conv2 = ChebConv(256, num_features, K=5)

    def forward(self, x, edge_index, train_edge_weight):
        x = F.relu(self.conv1(x, edge_index, train_edge_weight))
        x = self.conv2(x, edge_index, train_edge_weight)
        x = self.head(x)
        return x


class GMAEEG(nn.Module):
    """
    Graph Masked Autoencoder for EEG
    """

    def __init__(self, masked_num):
        super(GMAEEG, self).__init__()
        self.masked_num = masked_num


        self.encoder = Encoder(num_features=2496)
        self.encoder_to_decoder = Encoder_to_Decoder()
        self.decoder = Decoder(num_features=2496)

        self.smallconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )

        self.midconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )

        self.largeconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )

        self.smalldeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
        )

        self.middeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
        )

        self.largedeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 8), stride=(1, 2), padding=(0, 1)),
        )

        # mask and remask token
        self.enc_token = nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=(60))))
        self.dec_token = nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=(64))))

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)


    def forward(self, indata):
        # FOR loops instead of .reshape() are used,
        # to ensure operations are conducted on those dimensions with specific physical meanings
        x, edge_index, batch = indata['x'], indata['edge_index'], indata['batch_info']
        mask_node_list = random.sample(range(0, 32), self.masked_num)

        masked_x = x.clone()
        noise_x = x.clone()
        small_x = x.clone()
        mid_x = x.clone()
        large_x = x.clone()
        for index_batch in range(0, int((masked_x.shape[0])/32)):
            for index_mask in range(self.masked_num):
                dice_general = random.sample(range(0, 10), 1)
                dice_noise = random.sample(range(0, 32), 1)
                if dice_general[0] < 8:
                    masked_x[mask_node_list[index_mask] + 32 * index_batch, :] = self.enc_token
                elif dice_general[0] == 8:
                    masked_x[mask_node_list[index_mask] + 32 * index_batch, :] = noise_x[dice_noise[0] + 32 * index_batch, :]
                # dice_general[0] == 9: 'Unchanged'

        temp_x = torch.FloatTensor(np.zeros(shape=(int((x.shape[0])/32), 1, 32, 60))).cuda()
        for index_batch in range(0, int((x.shape[0]) / 32)):
            temp_x[index_batch, 0, :, :] = masked_x[index_batch*32:(index_batch+1)*32, :]


        smallconv_out = self.smallconv(temp_x)
        midconv_out = self.midconv(temp_x)
        largeconv_out = self.largeconv(temp_x)

        cat_conv_out = torch.cat((smallconv_out, midconv_out, largeconv_out), dim=3)

        per_conv_out = cat_conv_out.permute(0, 2, 1, 3)
        enc_in = torch.FloatTensor(np.zeros(shape=(int(32*(x.shape[0])/32), cat_conv_out.shape[1]*cat_conv_out.shape[3]))).cuda()

        for index_batch in range(0, int((x.shape[0]) / 32)):
            for index_channel in range(0, int(cat_conv_out.shape[3])):
                enc_in[index_batch*32:(index_batch+1)*32, index_channel*cat_conv_out.shape[1]:(index_channel+1)*cat_conv_out.shape[1]] = per_conv_out[index_batch, :, :, index_channel]


        out, _edge_weight, train_edge_weight = self.encoder(enc_in, edge_index)
        en_feature=out
        out = self.encoder_to_decoder(out)

        for index_batch in range(0, int((masked_x.shape[0])/32)):
            for index_mask in range(self.masked_num):
                out[mask_node_list[index_mask]+32*index_batch, :] = self.dec_token

        dec_out = self.decoder(out, edge_index, train_edge_weight)

        deconv_in = per_conv_out


        for index_batch in range(0, int((x.shape[0]) / 32)):
            for index_channel in range(0, int(cat_conv_out.shape[3])):
                deconv_in[index_batch, :, :, index_channel] = dec_out[index_batch * 32:(index_batch + 1) * 32,index_channel * cat_conv_out.shape[1]:(index_channel + 1) * cat_conv_out.shape[1]]
        per_deconv_in = deconv_in.permute(0, 2, 1, 3)

        small_deconv_in = per_deconv_in[:, :, :, 0:smallconv_out.shape[3]]
        mid_deconv_in = per_deconv_in[:, :, :, smallconv_out.shape[3]:smallconv_out.shape[3]+midconv_out.shape[3]]
        large_deconv_in = per_deconv_in[:, :, :, smallconv_out.shape[3]+midconv_out.shape[3]:per_deconv_in.shape[3]]

        smalldeconv_out = self.smalldeconv(small_deconv_in)
        middeconv_out = self.middeconv(mid_deconv_in)
        largedeconv_out = self.largedeconv(large_deconv_in)

        smallreshaped_out = small_x
        midreshaped_out = mid_x
        largereshaped_out = large_x

        for index_batch in range(0, int((x.shape[0]) / 32)):
            smallreshaped_out[index_batch * 32:(index_batch + 1) * 32, :] = smalldeconv_out[index_batch, 0, :, :]
            midreshaped_out[index_batch * 32:(index_batch + 1) * 32, :] = middeconv_out[index_batch, 0, :, :]
            largereshaped_out[index_batch * 32:(index_batch + 1) * 32, :] = largedeconv_out[index_batch, 0, :, :]


        smallreshaped_out = smallreshaped_out.unsqueeze(2)
        midreshaped_out = midreshaped_out.unsqueeze(2)
        largereshaped_out = largereshaped_out.unsqueeze(2)
        cat_reshaped_out = torch.cat((smallreshaped_out, midreshaped_out, largereshaped_out), dim=2)
        reshaped_out = torch.mean(cat_reshaped_out, dim=2)

        return reshaped_out, en_feature

class ConvBinaryClassifier(nn.Module):
    def __init__(self):
        super(ConvBinaryClassifier, self).__init__()

        # 1D 卷积提取时间特征（处理 64 维时间步）
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 保持维度不变
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # 降维到 16 维
        )

        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 输出1个值
        )

    def forward(self, x):
        x = x.view(-1, 32, 64)  # 重新调整形状: [batch_size, 32, 64]

        x = self.conv_layers(x)  # 经过 1D 卷积层
        x = x.view(x.shape[0], -1)  # 展平为 [batch_size, 128 * 16]

        x = self.classifier(x)  # 通过 MLP 分类
        return x  # 返回 logits

class ST(nn.Module):
    def __init__(self, opt):
        super(ST, self).__init__()
        #重建网络
        self.rebuild = GMAEEG(masked_num=3).cuda()

        #教师网络1
        self.teacher1=DGCNN(32,60,2).cuda()

        self.teacher2 = TCNClassifier(num_inputs=32, num_channels=[64, 128],
                              num_classes=2, kernel_size=3, dropout=0.2).cuda()
        # 学生网络
        self.student = EEGNet(num_channels=32, num_samples=60, num_classes=2,
                          dropoutRate=0.5, F1=8, D=2, F2=16, kernel_length=8)

        # self.teacher1 = CrossTransformer(num_inputs1, num_inputs2, chunk_len, hidden_dim, gpt_n_embd)
        #
        # self.teacher2 = DGCNN(num_inputs2, num_length, num_embed=gpt_n_embd)
        #重建分类损失
        self.re_filer=ConvBinaryClassifier()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_cls1=nn.BCEWithLogitsLoss
        self.criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
        self.criterion_div = DistillKL(opt['kd_T'])
        self.criterion_kd = CAMKD()
        self.criterion_recon = nn.MSELoss()
    def forward(self, x,y):


        #重建网络
        r_x= x.view(-1, x.size(2))
        r_input = {}
        # 672 = batch size (32) * EEG channels (21)         500 = slice time (2 seconds) * sample rate (250 Hz)
        r_input['x'] =r_x
        # 13440 = batch size (32) * number of non-zero elements in adj matrix (21 * 20)
        r_input['edge_index'] = torch.ones([2, x.size(0) * x.size(1)*(x.size(1)-1)], dtype=torch.int64).cuda()
        # 672 = batch size (32) * EEG channels (21)
        r_input['batch_info'] = torch.arange(0, x.size(0) * x.size(1)).repeat_interleave(32).cuda()
        recon_out,feature=self.rebuild(r_input)


        # logits_recon=self.re_filer(feature)


        loss_recon = self.criterion_recon(recon_out, r_x)
        # loss_cls=self.criterion_cls1(logits_recon,y)
        # loss_re = loss_recon + loss_cls  # 如有需要，可加权组合
        loss_re = loss_recon



        #教师网络1
        recon_out = recon_out.reshape(x.size(0), 32, 60)
        logits_t1=self.teacher1(recon_out)
        loss_t1=self.criterion_cls(logits_t1,y)

        #教师网络2
        logits_t2=self.teacher2(recon_out)
        loss_t2 = self.criterion_cls(logits_t2, y)

        #学生网络
        logits_s = self.student(recon_out)
        loss_s = self.criterion_cls(logits_s, y)

        # 蒸馏

        loss_t_list = [self.criterion_cls_lc(logits_t1, y), self.criterion_cls_lc(logits_t2, y)]
        loss_t = torch.stack(loss_t_list, dim=0)
        attention = (1.0 - F.softmax(loss_t, dim=0))
        loss_dist1_list1 = [
            self.criterion_div(logits_s, logits_t1, is_ca=True),
            self.criterion_div(logits_s, logits_t2, is_ca=True)
        ]
        loss_dist1 = torch.stack(loss_dist1_list1, dim=0)
        bsz1 = loss_dist1.shape[1]


        loss_dist1 = (torch.mul(attention, loss_dist1).sum()) / (1.0 * bsz1 * 2)




        # 自动根据模型模式确定 training 状态
        training = self.training
        if training:
            reversed_features = grad_reverse(logits_s, alpha=0.1)
            return logits_s, reversed_features, loss_re,loss_t1,loss_t2,loss_s,loss_dist1
        else:
            return logits_s