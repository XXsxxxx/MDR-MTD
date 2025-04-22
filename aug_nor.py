import torch
from sklearn.preprocessing import StandardScaler

# # 标准化函数
# def standardize_data(x1, x2):
#     """
#     对输入数据进行标准化。
    
#     参数:
#     - data: torch.Tensor, 数据的形状为 (N, C, F)，其中 N 是样本数，C 是通道数，F 是特征数。
    
#     返回:
#     - data_standardized: torch.Tensor, 标准化后的数据，形状与输入数据相同。
#     """
#     # Step 1: 在 N 维度上合并数据
#     combined_data = torch.cat((x1, x2), dim=0)  # 形状为 (N_hc + N_mdd, 128, 300)
#     # Step 2: 使用 Z-score 标准化，将数据范围缩放到 0~1
#     mean = combined_data.mean(dim=0, keepdim=True)
#     std = combined_data.std(dim=0, keepdim=True)
#     # 防止除以零的情况
#     std[std == 0] = 1
#     # Z-score 标准化
#     normalized_data = (combined_data - mean) / std
#     # Step 3: 恢复为 hc_data 和 MDD_data 的原始形状
#     HC_data = normalized_data[:x1.shape[0]]  # 形状为 (N_hc, 128, 300)
#     MDD_data = normalized_data[x1.shape[0]:]  # 形状为 (N_mdd, 128, 300)

#     return HC_data, MDD_data   # 恢复原始形状

# 标准化函数
def standardize_data(x):
    """
    对输入 4D 数据 (B, N, C, F) 或 3D 数据 (N, C, F) 进行标准化。
    
    参数:
    - x: torch.Tensor, 输入数据。形状为 (B, N, C, F) 或 (N, C, F)。
    
    返回:
    - x_standardized: torch.Tensor, 标准化后的数据，形状与输入相同。
    """
    if x.dim() == 3:  # 如果是 3D 数据
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std[std == 0] = 1  # 防止除以零
        return (x - mean) / std
    elif x.dim() == 4:  # 如果是 4D 数据
        mean = x.mean(dim=1, keepdim=True)  # 对 N 维度 (样本维度) 求均值
        std = x.std(dim=1, keepdim=True)    # 对 N 维度求标准差
        std[std == 0] = 1  # 防止除以零
        return (x - mean) / std
    else:
        raise ValueError("输入数据的维度必须是 3D 或 4D!")

def augment_data(data, labels, times=2):
    """
    对数据进行增强，支持 4D 数据 (B, N, C, F) 或 3D 数据 (N, C, F)。
    
    参数:
    - data: torch.Tensor, 数据，形状为 (B, N, C, F) 或 (N, C, F)。
    - labels: torch.Tensor, 标签，形状为 (B, N, 1) 或 (N, 1)。
    - times: int, 增加样本的倍数。
    
    返回:
    - augmented_data: torch.Tensor, 增强后的数据，形状为 (B, N*times, C, F) 或 (N*times, C, F)。
    - augmented_labels: torch.Tensor, 增强后的标签，形状为 (B, N*times, 1) 或 (N*times, 1)。
    """
    if data.dim() == 3:  # 3D 数据处理
        augmented_data = data.clone()
        augmented_labels = labels.clone()
        for _ in range(times - 1):
            noise = torch.randn_like(data) * 0.01
            augmented_data = torch.cat((augmented_data, data + noise), dim=0)
            augmented_labels = torch.cat((augmented_labels, labels), dim=0)
        return augmented_data, augmented_labels

    elif data.dim() == 4:  # 4D 数据处理
        B, N, C, F = data.shape
        augmented_data = data.clone()
        augmented_labels = labels.clone()
        for _ in range(times - 1):
            noise = torch.randn_like(data) * 0.01
            augmented_data = torch.cat((augmented_data, data + noise), dim=0)  # 在样本维度扩展
            augmented_labels = torch.cat((augmented_labels, labels), dim=0)   # 在样本维度扩展
        return augmented_data, augmented_labels

    else:
        raise ValueError("输入数据的维度必须是 3D 或 4D!")

