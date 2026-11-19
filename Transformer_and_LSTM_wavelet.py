#!/usr/bin/env python3
"""
================================================================================
Transformer风速预报模型 —— 10min采样，输入/输出时长均可变 (1~24h)
================================================================================
【功能概述】
1. 任务：利用过去 input_hours 小时风速序列，预测未来 output_hours 小时风速序列；
   X序列：过去 input_hours 小时，包含起报时刻，共 input_hours*6+1 点；
   Y序列：未来 output_hours 小时，包含起报时刻，共 output_hours*6+1 点；
   起报时刻在X末尾、Y开头。
2. 数据源目录 wavelet_10min（db8 小波分解，D1~D10 细节 + A 近似，完美重构无残差），目录结构：
    wavelet_10min/
        F111/
            F111_wavelet_decomposition.txt
   ✅ 自动跳过风机 F125。
3. 支持模态：raw / d1~d9 / a / cum1~cum10 / high_freq / mid_freq / low_freq
   （d1~d9 为细节分量，a 为近似分量；cumN 为后 N 个低频分量累加；
    high_freq=D1-D2, mid_freq=D3-D5, low_freq=D6-A）
4. 地形分组：山上 hill (F411-F436, F541-F563) / 山下 low (F111-F143, F251-F286)
5. Dataset逻辑：在线滑动生成样本，缓存有效起报时刻
6. 滑动步长：train=1(10min)，val/test=6(60min)
7. 归一化：输入 Z-score，输出 Min-Max [0,1]
8. 模型：Transformer编码器 + 时间编码 + 绝对时间编码
9. 训练策略：早停(验证集RMSE)，保存val-best模型

【关键命令行参数】
--mod              选择模态：cum7 / raw / cum9 / high_freq ...
--gpu              指定GPU卡号
--input_hours      输入序列覆盖的过去小时数（1~24），默认4
--output_hours     预测的未来小时数（1~24），默认4
--terrain          统一设置 train/val/test 地形: hill/low/all
--terrain_train    训练集地形，优先级高于 --terrain
--terrain_val      val集地形
--terrain_test     test集地形
--fan_id           指定单个风机，如 F111
--no_station_time  关闭相对时间编码
--no_absolute_time 关闭绝对时间编码
--refresh_cache    强制重建样本缓存

【运行示例】
# 输入4小时，预测1小时（山上训练，山上测试）
python Transformer.py --mod cum7 --gpu 0 --output_hours 1 --terrain hill --refresh_cache

# 输入8小时，预测24小时（跨地形：山上训练，山下测试）
python Transformer.py --mod cum7 --gpu 0 --input_hours 8 --output_hours 24 --terrain_train hill --terrain_test low --refresh_cache

# 单风机调试，启用时间编码，重建缓存，使用所有风机
python Transformer.py --mod raw --gpu 0 --fan_id F111 --refresh_cache
================================================================================
"""

import os
import sys
# 手动从命令行参数中提取 --gpu 的值
def _set_gpu_env():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_ids = sys.argv[i+1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"[预加载] 已设置 CUDA_VISIBLE_DEVICES={gpu_ids}")
            return

_set_gpu_env()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from datetime import datetime, timedelta
import argparse
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 配置类 Config
# =============================================================================
class Config:
    # 时序参数：10min采样
    dt_minute = 10

    # 滑动步长：单位=样本点，1点=10min
    slide_step_train = 1    # 训练集每10min一个样本
    slide_step_eval = 6     # val/test每60min一个样本

    # 输出目标 min‑max
    target_min = 0.0
    target_max = 25.0

    # 模型超参
    hidden_dim = 384
    lstm_hidden_size = 512  # LSTM专用隐藏层维度
    model_type = "transformer"  # transformer / lstm / hybrid
    fuse_dim = 384  # hybrid模式下的融合特征维度

    num_layers = 4
    num_heads = 8
    dropout = 0.35
    learning_rate = 2e-4
    weight_decay = 1e-5
    batch_size = 256
    num_epochs = 100

    # loss参数
#    lambda_first = 3.0

    # 模态
    mod = "cum7"

    # 时间编码开关
    use_station_time = True
    use_absolute_time = True

    # -------- 路径配置 --------
    data_dir = "../wavelet10_10min"
    cache_dir = "./cache"
    output_dir = "./output"
    checkpoint_dir = "./checkpoints"

    # -------- 时间区间 --------
    train_start = datetime.fromisoformat("2023-07-21 00:00:00")
    train_end   = datetime.fromisoformat("2025-02-01 00:00:00")
    val_start   = datetime.fromisoformat("2025-02-01 00:10:00")
    val_end     = datetime.fromisoformat("2025-04-20 00:00:00")
    test_start  = datetime.fromisoformat("2025-04-20 00:10:00")
    test_end    = datetime.fromisoformat("2025-07-19 23:50:00")

    # 早停参数
    early_stopping_patience = 15
    early_stopping_min_delta = 1e-5
    early_stopping_mode = "min"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 输入/输出时长（小时） ==========
    input_hours = 4   # 默认4小时
    output_hours = 4  # 默认4小时


# =============================================================================
# 模态映射
# =============================================================================
NAME_MAPPING = {
    '原始序列': 'raw',
    'D1': 'd1',
    'D2': 'd2',
    'D3': 'd3',
    'D4': 'd4',
    'D5': 'd5',
    'D6': 'd6',
    'D7': 'd7',
    'D8': 'd8',
    'D9': 'd9',
    'D10': 'd10',
    'A': 'a',
    '后1个累加': 'cum1',
    '后2个累加': 'cum2',
    '后3个累加': 'cum3',
    '后4个累加': 'cum4',
    '后5个累加': 'cum5',
    '后6个累加': 'cum6',
    '后7个累加': 'cum7',
    '后8个累加': 'cum8',
    '后9个累加': 'cum9',
    '后10个累加': 'cum10',
    '高频(D1-D2)': 'high_freq',
    '中频(D3-D6)': 'mid_freq',
    '低频(D7-A)': 'low_freq',
}
MOD_TO_COLNAME = {v: k for k, v in NAME_MAPPING.items()}

# ============地形风机分组定义 ============
HILL_FID_SET = {
    "411","412","413","414","415","416",
    "421","422","423","424","425","426",
    "431","432","433","434","435","436",
    "541","542","543","544","545","546",
    "551","552","553",
    "561","562","563"
}
LOW_FID_SET = {
    "111","112","113","114",
    "121","122","123","124","125","126",
    "131","132","133","134","135","136",
    "141","142","143",
    "251","252","253","254",
    "261","262","263","264","265","266",
    "271","272","273","274","275",
    "281","282","283","284","285","286"
}
TERRAIN_MAP = {
    "hill": HILL_FID_SET,
    "low": LOW_FID_SET,
    "all": None
}


# =============================================================================
# 早停类
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"  📊 初始最佳指标: {score:.4f}")
            return False
        if self.mode == 'min':
            improvement = self.best_score - score
            is_improved = improvement > self.min_delta
        else:
            improvement = score - self.best_score
            is_improved = improvement > self.min_delta

        if is_improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"  ✅ 指标改善: {score:.4f} (最佳: {self.best_score:.4f})")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️ 指标未改善 ({self.counter}/{self.patience}), 当前: {score:.4f}, 最佳: {self.best_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  🛑 早停触发！连续 {self.patience} 个epoch没有改善")
                return True
            return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# =============================================================================
# 工具函数
# =============================================================================
def load_fan_series_multicol(filepath, mod: str):
    """返回包含 dt, speed (模态列), raw_series (原始序列) 的 DataFrame
    小波分解为完美重构（sum(D1~D10+A)=原始序列），无残差概念，故不支持 res 模态。
    """
    with open(filepath, 'r', encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    if len(lines) < 2:
        return pd.DataFrame({"dt": [], "speed": [], "raw_series": []})

    header_line = lines[0]
    header = [h.strip() for h in header_line.split("\t")]
    raw_col = '原始序列'
    if raw_col not in header:
        raise KeyError(f"表头不存在列:{raw_col}；现有表头:{header}")
    raw_idx = header.index(raw_col)

    # ── 普通模态：直接读取对应列 ──
    if mod not in MOD_TO_COLNAME:
        raise KeyError(f"不支持的模态:{mod}；可选: {sorted(MOD_TO_COLNAME.keys())}")
    col_target = MOD_TO_COLNAME[mod]
    if col_target not in header:
        raise KeyError(f"表头不存在列:{col_target}；现有表头:{header}")
    target_idx = header.index(col_target)

    dates = []
    speed_vals = []
    raw_vals = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2:
            continue
        try:
            dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = pd.NaT
        try:
            val = float(parts[target_idx])
        except (ValueError, IndexError):
            val = np.nan
        try:
            raw_val = float(parts[raw_idx])
        except (ValueError, IndexError):
            raw_val = np.nan
        dates.append(dt)
        speed_vals.append(val)
        raw_vals.append(raw_val)

    df = pd.DataFrame({"dt": dates, "speed": speed_vals, "raw_series": raw_vals})
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["raw_series"] = pd.to_numeric(df["raw_series"], errors="coerce")
    return df


def get_time_encoding(t0: datetime, seq_len: int, pred_len: int):
    """
    相对时间编码、绝对时间编码
    X部分 offset [-(seq_len-1)*10, ..., 0] 步长10
    Y部分 offset [0,10,...,(pred_len-1)*10]
    """
    past_offsets = np.arange(-(seq_len - 1) * 10, 10, 10)
    fut_offsets = np.arange(10, (pred_len+1)*10, 10)

    past_rel = past_offsets / 240.0
    fut_rel = fut_offsets / 240.0

    past_time_sin = np.sin(2 * np.pi * past_rel)
    past_time_cos = np.cos(2 * np.pi * past_rel)
    past_time_feat = np.stack([past_time_sin, past_time_cos], axis=1)

    future_time_sin = np.sin(2 * np.pi * fut_rel)
    future_time_cos = np.cos(2 * np.pi * fut_rel)
    future_time_feat = np.stack([future_time_sin, future_time_cos], axis=1)

    # 绝对时间编码（8维）
    month_sin = np.sin(2 * np.pi * t0.month / 12)
    month_cos = np.cos(2 * np.pi * t0.month / 12)
    doy = t0.timetuple().tm_yday
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    hour = t0.hour + t0.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_of_day = t0.hour * 6 + t0.minute // 10
    minute_sin = np.sin(2 * np.pi * minute_of_day / 144)
    minute_cos = np.cos(2 * np.pi * minute_of_day / 144)

    absolute_time = np.array([
        month_sin, month_cos,
        doy_sin, doy_cos,
        hour_sin, hour_cos,
        minute_sin, minute_cos
    ], dtype=np.float32)

    return past_time_feat, future_time_feat, absolute_time


# =============================================================================
# 评估指标函数
# =============================================================================
def calculate_metrics(predictions, targets, config):
    valid_mask = np.isfinite(targets) & np.isfinite(predictions)
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    if len(valid_preds) == 0:
        return {
            'mse': np.nan, 'rmse': np.nan, 'mae': np.nan,
            'smape': np.nan, 'r2': np.nan, 'bias': np.nan,
            'n_samples': 0
        }

    mse = np.mean((valid_preds - valid_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(valid_preds - valid_targets))
    denom = (np.abs(valid_targets) + np.abs(valid_preds) + 1e-8)
    smape = np.mean(2 * np.abs(valid_targets - valid_preds) / denom) * 100
    ss_res = np.sum((valid_targets - valid_preds) ** 2)
    ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    bias = np.mean(valid_preds - valid_targets)

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'smape': smape, 'r2': r2, 'bias': bias,
        'n_samples': len(valid_preds)
    }


def calculate_step_metrics(predictions, targets, pred_len, config):
    step_metrics = {
        'step_rmse': [], 'step_mae': [], 'step_smape': [],
        'step_r2': [], 'step_bias': []
    }
    for i in range(pred_len):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        valid_mask = np.isfinite(target_i) & np.isfinite(pred_i)
        pred_valid = pred_i[valid_mask]
        target_valid = target_i[valid_mask]
        if len(pred_valid) == 0:
            for k in step_metrics:
                step_metrics[k].append(np.nan)
            continue
        rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
        mae = np.mean(np.abs(pred_valid - target_valid))
        denom = (np.abs(target_valid) + np.abs(pred_valid) + 1e-8)
        smape = np.mean(2 * np.abs(target_valid - pred_valid) / denom) * 100
        ss_res = np.sum((target_valid - pred_valid) ** 2)
        ss_tot = np.sum((target_valid - np.mean(target_valid)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        bias = np.mean(pred_valid - target_valid)
        step_metrics['step_rmse'].append(rmse)
        step_metrics['step_mae'].append(mae)
        step_metrics['step_smape'].append(smape)
        step_metrics['step_r2'].append(r2)
        step_metrics['step_bias'].append(bias)
    return step_metrics


def print_metrics(metrics, title="评估指标"):
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"有效样本数: {metrics['n_samples']}")
    print(f"MSE:  {metrics['mse']:.6f} (m/s)²")
    print(f"RMSE: {metrics['rmse']:.6f} m/s")
    print(f"MAE:  {metrics['mae']:.6f} m/s")
    print(f"sMAPE: {metrics['smape']:.2f} %")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Bias: {metrics['bias']:.6f} m/s")


def print_step_metrics(step_metrics, pred_len):
    print(f"\n{'='*60}")
    print("📊 各时间步评估指标")
    print(f"{'='*60}")
    print(f"{'步数':>6} | {'分钟':>6} | {'RMSE':>10} | {'MAE':>10} | {'sMAPE':>10} | {'R²':>10} | {'Bias':>10}")
    print("-"*75)
    for i in range(pred_len):
        minutes = i * 10
        rmse = step_metrics['step_rmse'][i] if i < len(step_metrics['step_rmse']) else np.nan
        mae = step_metrics['step_mae'][i] if i < len(step_metrics['step_mae']) else np.nan
        smape = step_metrics['step_smape'][i] if i < len(step_metrics['step_smape']) else np.nan
        r2 = step_metrics['step_r2'][i] if i < len(step_metrics['step_r2']) else np.nan
        bias = step_metrics['step_bias'][i] if i < len(step_metrics['step_bias']) else np.nan
        if not np.isnan(rmse):
            print(f"{i:>6} | {minutes:>6} | {rmse:>10.4f} | {mae:>10.4f} | {smape:>10.2f} | {r2:>10.4f} | {bias:>10.4f}")
        else:
            print(f"{i:>6} | {minutes:>6} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")


def plot_training_history(history, checkpoint_dir, mod_name):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(checkpoint_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title(f'{mod_name} - Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    axes[0, 1].plot(epochs, history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('RMSE (m/s)')
    axes[0,1].set_title(f'{mod_name} - RMSE'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 2].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[0,2].set_xlabel('Epoch'); axes[0,2].set_ylabel('MAE (m/s)')
    axes[0,2].set_title(f'{mod_name} - MAE'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_smape'], 'b-', label='Train sMAPE', linewidth=2)
    axes[1, 0].plot(epochs, history['val_smape'], 'r-', label='Val sMAPE', linewidth=2)
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('sMAPE (%)')
    axes[1,0].set_title(f'{mod_name} - sMAPE'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=2)
    axes[1, 1].plot(epochs, history['val_r2'], 'r-', label='Val R²', linewidth=2)
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('R²')
    axes[1,1].set_title(f'{mod_name} - R²'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history['train_bias'], 'b-', label='Train Bias', linewidth=2)
    axes[1, 2].plot(epochs, history['val_bias'], 'r-', label='Val Bias', linewidth=2)
    axes[1,2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,2].set_xlabel('Epoch'); axes[1,2].set_ylabel('Bias (m/s)')
    axes[1,2].set_title(f'{mod_name} - Bias'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'), dpi=300)
    plt.close()
    print(f"\n📊 训练历史曲线已保存: {os.path.join(checkpoint_dir, 'training_history.png')}")


def plot_all_epochs_performance(metrics_summary, config, checkpoint_dir=None):
    import matplotlib.pyplot as plt
    import matplotlib
    if not metrics_summary:
        return
    matplotlib.use('Agg')
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.checkpoint_dir, config.mod)
    plot_path = os.path.join(checkpoint_dir, f'all_epochs_performance_{config.mod}.png')
    epochs = [item['epoch'] for item in metrics_summary]
    rmse = [item['rmse'] for item in metrics_summary]
    mae = [item['mae'] for item in metrics_summary]
    smape = [item['smape'] for item in metrics_summary]
    r2 = [item['r2'] for item in metrics_summary]
    bias = [item['bias'] for item in metrics_summary]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0,0].plot(epochs, rmse, 'r-o', linewidth=2)
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("RMSE (m/s)")
    axes[0,0].set_title("RMSE by Epoch"); axes[0,0].grid(True, alpha=0.3)
    best_idx = np.argmin(rmse)
    axes[0,0].plot(epochs[best_idx], rmse[best_idx], 'g*', markersize=15)

    axes[0,1].plot(epochs, mae, 'b-s', linewidth=2)
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("MAE (m/s)")
    axes[0,1].set_title("MAE by Epoch"); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(epochs, smape, 'g-^', linewidth=2)
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("sMAPE (%)")
    axes[0,2].set_title("sMAPE by Epoch"); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(epochs, r2, 'm-d', linewidth=2)
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("R²")
    axes[1,0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
    axes[1,0].set_title("R² by Epoch"); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, bias, 'c-v', linewidth=2)
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Bias (m/s)")
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_title("Bias by Epoch"); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].scatter(mae, rmse, c=epochs, cmap='viridis', s=80, alpha=0.7)
    axes[1,2].set_xlabel("MAE"); axes[1,2].set_ylabel("RMSE")
    axes[1,2].set_title("RMSE vs MAE"); axes[1,2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n📊 全epoch性能对比图已保存:{plot_path}")


# =============================================================================
# Transformer模型组件
# =============================================================================
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=384, depth=4, num_heads=8, dropout=0.35):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x)
        x = x + self.pos_embed[:, :T]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class WindTransformer(nn.Module):
    def __init__(self, config):
        super(WindTransformer, self).__init__()
        self.config = config
        
        input_dim = 1
        if config.use_station_time:
            input_dim += 2
        
        self.encoder = TemporalTransformer(
            input_dim=input_dim,
            embed_dim=config.hidden_dim,
            depth=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        if config.use_absolute_time:
            self.absolute_time_proj = nn.Sequential(
                nn.Linear(8, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, config.hidden_dim)
            )
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.pred_len)

    def forward(self, x, station_time, absolute_time):
        if self.config.use_station_time:
            encoder_input = torch.cat([x, station_time], dim=-1)
        else:
            encoder_input = x
        
        feat = self.encoder(encoder_input)
        
        if self.config.use_absolute_time:
            abs_time_emb = self.absolute_time_proj(absolute_time)
            abs_time_emb = abs_time_emb.unsqueeze(1)
            feat = feat + abs_time_emb
        
        x = feat[:, -1, :]
        x = self.norm(x)
        output = self.head(x)
        
        return output


class WindLSTM(nn.Module):
    """
    多层LSTM编码器 + 全连接解码器
    接口与WindTransformer一致：输入(B, seq_len, 1) → 输出(B, pred_len)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_size = 1
        self.use_station_time = config.use_station_time
        self.use_absolute_time = config.use_absolute_time
        if config.use_station_time:
            input_size += 2

        self.abs_time_dim = 8
        if config.use_absolute_time:
            self.fc_time = nn.Linear(self.abs_time_dim, config.lstm_hidden_size // 4)

        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=False
        )

        decode_in_dim = config.lstm_hidden_size
        if config.use_absolute_time:
            decode_in_dim += config.lstm_hidden_size // 4

        self.fc_decoder = nn.Sequential(
            nn.Linear(decode_in_dim, config.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size, config.pred_len)
        )

    def forward(self, x, station_time=None, absolute_time=None):
        # x: (B, seq_len, 1) → LSTM需要(seq_len, B, feat)
        x = x.permute(1, 0, 2)
        if self.use_station_time and station_time is not None:
            st = station_time.permute(1, 0, 2)  # (seq_len, B, 2)
            lstm_input = torch.cat([x, st], dim=2)
        else:
            lstm_input = x
        encoder_out, (h_n, c_n) = self.lstm_encoder(lstm_input)
        last_hidden = h_n[-1, :, :]  # (B, hidden)
        if self.use_absolute_time and absolute_time is not None:
            time_feat = self.fc_time(absolute_time)
            feat_combine = torch.cat([last_hidden, time_feat], dim=1)
        else:
            feat_combine = last_hidden
        pred = self.fc_decoder(feat_combine)  # (B, pred_len)
        return pred

class WindHybrid(nn.Module):
    """
    Transformer-LSTM 混合预测器（门控融合）：
    - Transformer 分支：捕捉全局长程依赖（中低频周期）
    - LSTM 分支：捕捉局部时序动态（高频惯性、梯度变化）
    - 可学习门控：自适应加权两个分支的特征，低频时Transformer主导，高频时LSTM补充
    接口与 WindTransformer 一致：输入(B, seq_len, 1) → 输出(B, pred_len)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_station_time = config.use_station_time
        self.use_absolute_time = config.use_absolute_time
        fuse_dim = config.fuse_dim

        # ---------- Transformer 分支 ----------
        t_input_dim = 1 + (2 if config.use_station_time else 0)
        self.t_encoder = TemporalTransformer(
            input_dim=t_input_dim,
            embed_dim=config.hidden_dim,
            depth=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        if config.use_absolute_time:
            self.t_abs_proj = nn.Sequential(
                nn.Linear(8, config.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 4, config.hidden_dim)
            )
        self.t_norm = nn.LayerNorm(config.hidden_dim)

        # ---------- LSTM 分支 ----------
        l_input_size = 1 + (2 if config.use_station_time else 0)
        if config.use_absolute_time:
            self.l_fc_time = nn.Linear(8, config.lstm_hidden_size // 4)
        self.lstm_encoder = nn.LSTM(
            input_size=l_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=False
        )

        # ---------- 特征投影到统一融合维度 ----------
        self.t_proj = nn.Linear(config.hidden_dim, fuse_dim)
        lstm_feat_dim = config.lstm_hidden_size + (config.lstm_hidden_size // 4 if config.use_absolute_time else 0)
        self.l_proj = nn.Linear(lstm_feat_dim, fuse_dim)

        # ---------- 门控融合 ----------
        self.gate = nn.Sequential(
            nn.Linear(fuse_dim * 2, fuse_dim),
            nn.Sigmoid()
        )
        self.fuse_norm = nn.LayerNorm(fuse_dim)
        self.head = nn.Linear(fuse_dim, config.pred_len)

    def forward(self, x, station_time=None, absolute_time=None):
        # === Transformer 分支：取最后一步特征 ===
        if self.use_station_time and station_time is not None:
            t_input = torch.cat([x, station_time], dim=-1)
        else:
            t_input = x
        t_feat = self.t_encoder(t_input)
        if self.use_absolute_time and absolute_time is not None:
            t_feat = t_feat + self.t_abs_proj(absolute_time).unsqueeze(1)
        t_feat = self.t_norm(t_feat[:, -1, :])  # (B, hidden_dim)

        # === LSTM 分支：取最后一层隐状态 + 绝对时间 ===
        x_lstm = x.permute(1, 0, 2)
        if self.use_station_time and station_time is not None:
            lstm_input = torch.cat([x_lstm, station_time.permute(1, 0, 2)], dim=2)
        else:
            lstm_input = x_lstm
        _, (h_n, _) = self.lstm_encoder(lstm_input)
        l_last = h_n[-1, :, :]  # (B, lstm_hidden)
        if self.use_absolute_time and absolute_time is not None:
            l_feat = torch.cat([l_last, self.l_fc_time(absolute_time)], dim=1)
        else:
            l_feat = l_last

        # === 门控融合 ===
        t_proj = self.t_proj(t_feat)   # (B, fuse_dim)
        l_proj = self.l_proj(l_feat)   # (B, fuse_dim)
        gate = self.gate(torch.cat([t_proj, l_proj], dim=-1))  # (B, fuse_dim), 值∈(0,1)
        fused = gate * t_proj + (1.0 - gate) * l_proj
        fused = self.fuse_norm(fused)
        output = self.head(fused)  # (B, pred_len)
        return output


def weighted_loss(output, target, lambda_first=None):
    """已移除首步加权，lambda_first参数保留仅为调用兼容，不再使用"""
    if output.dim() > 2:
        output = output.squeeze(-1)
    if target.dim() > 2:
        target = target.squeeze(-1)
    if output.shape != target.shape:
        raise ValueError(f"Shape mismatch: output {output.shape}, target {target.shape}")
    mse = (output - target) ** 2
    return mse.mean()

# =============================================================================
# Dataset：在线滑动完整风机序列（支持可变输入/输出长度 + 地形过滤）
# =============================================================================
class WindFullSeqDataset(Dataset):
    def __init__(self, config: Config, mode: str, norm_stats=None, refresh_cache=False,
                 select_fan_id="all", terrain="all"):
        self.select_fan_id = select_fan_id
        self.terrain = terrain
        self.config = config
        self.mode = mode
        self.mod = config.mod
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.use_station_time = config.use_station_time
        self.use_absolute_time = config.use_absolute_time

        if mode == "train":
            self.t_start = config.train_start
            self.t_end = config.train_end
            self.slide_step = config.slide_step_train
        elif mode == "val":
            self.t_start = config.val_start
            self.t_end = config.val_end
            self.slide_step = config.slide_step_eval
        elif mode == "test":
            self.t_start = config.test_start
            self.t_end = config.test_end
            self.slide_step = config.slide_step_eval
        else:
            raise ValueError("mode must be train / val / test")

        os.makedirs(config.cache_dir, exist_ok=True)
        cache_file = os.path.join(
            config.cache_dir,
            f"cache_{mode}_{self.mod}_{self.select_fan_id}_in{config.input_hours}h_out{config.output_hours}h_ter{self.terrain}.json"
        )

        cache_meta = {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "slide_step": self.slide_step,
            "t_start": self.t_start.isoformat(),
            "t_end": self.t_end.isoformat(),
            "mod": self.mod,
            "input_hours": config.input_hours,
            "output_hours": config.output_hours,
            "terrain": self.terrain,
            "fan_id": self.select_fan_id
        }

        # 加载风机文件并应用地形过滤
        fan_files = []
        for subdir_name in os.listdir(config.data_dir):
            subdir_full = os.path.join(config.data_dir, subdir_name)
            if not os.path.isdir(subdir_full):
                continue
            m_dir = re.match(r"^F(\d+)$", subdir_name)
            if not m_dir:
                continue
            fid = m_dir.group(1)
            if fid == "125":
                print(f"⚠️ 跳过风机 F{fid}")
                continue

            if self.select_fan_id != "all":
                m_sel = re.match(r"^F(\d+)$", self.select_fan_id.strip())
                if m_sel:
                    sel_num = m_sel.group(1)
                    if fid != sel_num:
                        continue
                else:
                    raise ValueError(f"--fan_id 格式错误，请使用 F111 格式")
            else:
                if self.terrain not in TERRAIN_MAP:
                    raise ValueError(f"terrain 必须是 hill / low / all")
                allow_set = TERRAIN_MAP[self.terrain]
                if allow_set is not None and fid not in allow_set:
                    continue

            fname = f"F{fid}_wavelet_decomposition.txt"
            fp = os.path.join(subdir_full, fname)
            if os.path.isfile(fp):
                fan_files.append((fid, fp))

        if len(fan_files) == 0:
            raise FileNotFoundError(f"{config.data_dir} 下没有找到符合条件风机！")

        self.fan_data = {}
        for fid, fp in fan_files:
            df = load_fan_series_multicol(fp, mod=self.mod)
            self.fan_data[fid] = df
            print(f"==== F{fid} 原始文件时间范围 ====")
            print(f"min dt: {df['dt'].min()}")
            print(f"max dt: {df['dt'].max()}")
            print(f"总行数:{len(df)}")
            if pd.isna(df['dt'].min()):
                print("!!!警告：dt全部是NaT，时间解析失败！")

        # 缓存逻辑
        cache_valid = False
        if (not refresh_cache) and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded["meta"] == cache_meta:
                self.sample_records = loaded["samples"]
                cache_valid = True

        if not cache_valid:
            print(f"重建 {mode} 样本缓存 (input={config.input_hours}h, output={config.output_hours}h, terrain={self.terrain}) ...")
            self.sample_records = []
            # X包含t0，Y也包含t0，总窗口 = seq_len + pred_len - 1
            total_win = self.seq_len + self.pred_len
            for fid, df in self.fan_data.items():
                n = len(df)
                max_s = n - total_win
                if max_s <= 0:
                    print(f"DEBUG fid={fid} skip: max_s={max_s}, need >=0")
                    continue
                for s_idx in range(0, max_s + 1, self.slide_step):
                    sub = df.iloc[s_idx: s_idx + total_win]
                    if sub["speed"].isna().any() or sub["raw_series"].isna().any():
                        continue
                    t0 = sub.iloc[self.seq_len - 1]["dt"]
                    if not (pd.Timestamp(self.t_start) <= pd.Timestamp(t0) <= pd.Timestamp(self.t_end)):
                        continue
                    self.sample_records.append({"fan_id": fid, "start_idx": int(s_idx), "t0": t0.isoformat()})
            out = {"meta": cache_meta, "samples": self.sample_records}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"{mode} 数据集有效样本数: {len(self.sample_records)}")

        # 训练集统计量（基于输入 X 列）
        if mode == "train":
            all_sp = []
            for fid, df in self.fan_data.items():
                df_tr = df[(df["dt"] >= config.train_start) & (df["dt"] <= config.train_end)]
                sp_arr = df_tr["speed"].dropna().values
                all_sp.extend(sp_arr.tolist())
            arr = np.array(all_sp, dtype=np.float32)
            self.norm_stats = {
                "x_mean": float(np.mean(arr)),
                "x_std": float(np.std(arr)),
                "y_min": config.target_min,
                "y_max": config.target_max
            }
        else:
            self.norm_stats = norm_stats

        self.x_mean = self.norm_stats["x_mean"]
        self.x_std = self.norm_stats["x_std"]
        self.y_min = self.norm_stats["y_min"]
        self.y_max = self.norm_stats["y_max"]

    def __len__(self):
        return len(self.sample_records)

    def __getitem__(self, idx):
        rec = self.sample_records[idx]
        fid = rec["fan_id"]
        s_idx = rec["start_idx"]
        t0 = datetime.fromisoformat(rec["t0"])
        df = self.fan_data[fid]
        total_win = self.seq_len + self.pred_len
        sub = df.iloc[s_idx: s_idx + total_win]

        # 修改后
        # X不变：0~seq_len‑1，包含t0
        x_raw = sub["speed"].values[:self.seq_len].astype(np.float32)
        # Y：t0的**下一个点开始**，取pred_len=24个点，不含t0
        y_raw = sub["speed"].values[self.seq_len : self.seq_len + self.pred_len].astype(np.float32)  # Y也取模态列
        x_norm = (x_raw - self.x_mean) / (self.x_std + 1e-8)
        y_norm = (y_raw - self.x_mean) / (self.x_std + 1e-8)  # 与X共用Z-score统计量

        past_time_feat, future_time_feat, absolute_time = get_time_encoding(t0, self.seq_len, self.pred_len)

        station = torch.from_numpy(x_norm).unsqueeze(-1).float()
        station_time = torch.from_numpy(past_time_feat).float()
        target = torch.from_numpy(y_norm).float()
        x_original = torch.from_numpy(x_raw).float()
        y_original = torch.from_numpy(y_raw).float()

        return {
            "station": station,
            "station_time": station_time,
            "absolute_time": torch.from_numpy(absolute_time).float(),
            "target": target,
            "timestamp": rec["t0"],
            "fan_id": fid,
            "x_original": x_original,
            "y_original": y_original
        }

    def print_sample(self, idx, label="样本"):
        if idx >= len(self.sample_records):
            print(f"⚠️ 索引 {idx} 超出范围（共 {len(self.sample_records)} 个样本）")
            return

        rec = self.sample_records[idx]
        fid = rec["fan_id"]
        s_idx = rec["start_idx"]
        t0 = datetime.fromisoformat(rec["t0"])
        df = self.fan_data[fid]

        total_win = self.seq_len + self.pred_len
        sub = df.iloc[s_idx: s_idx + total_win]

        x_raw = sub["speed"].values[:self.seq_len].astype(np.float32)
        y_raw = sub["speed"].values[self.seq_len : self.seq_len + self.pred_len].astype(np.float32)

        print(f"\n🔍 {label} 样本示例 (索引 {idx}):")
        print(f"  风机: {fid}")
        print(f"  起报时刻 t0: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  输入 X (模态列, {self.seq_len} 个点, 过去 {self.config.input_hours} 小时):")
        print(f"    {x_raw} ... (共 {len(x_raw)} 点)")
        print(f"  输出 Y (模态列, 与X同模态, {self.pred_len} 个点, 未来 {self.config.output_hours} 小时):")
        print(f"    {y_raw} ... (共 {len(y_raw)} 点)")
        print(f"  X 均值: {x_raw.mean():.3f}, 标准差: {x_raw.std():.3f}")
        print(f"  Y 均值: {y_raw.mean():.3f}, 标准差: {y_raw.std():.3f}")
        print("  " + "-"*40)


# =============================================================================
# 训练主循环
# =============================================================================
def train_loop(cfg,
               refresh_cache=False,
               select_fan_id="all",
               terrain_train="all",
               terrain_val="all",
               terrain_test="all"):
    ds_train = WindFullSeqDataset(cfg, mode="train", refresh_cache=refresh_cache,
                                  select_fan_id=select_fan_id, terrain=terrain_train)
    norm_stats = ds_train.norm_stats
    ds_val = WindFullSeqDataset(cfg, mode="val", norm_stats=norm_stats, refresh_cache=False,
                                select_fan_id=select_fan_id, terrain=terrain_val)
    ds_test = WindFullSeqDataset(cfg, mode="test", norm_stats=norm_stats, refresh_cache=False,
                                 select_fan_id=select_fan_id, terrain=terrain_test)

    if len(ds_train) == 0:
        raise ValueError("训练集为空！请检查时间范围和模态配置。")
    if len(ds_val) == 0:
        print("⚠️ 警告: 验证集为空！跳过验证。")
    if len(ds_test) == 0:
        print("⚠️ 警告: 测试集为空！跳过测试。")

    # 打印样本示例
    if len(ds_train) > 0:
        ds_train.print_sample(0, "训练集")
    if len(ds_val) > 0:
        ds_val.print_sample(0, "验证集")
    if len(ds_test) > 0:
        ds_test.print_sample(0, "测试集")

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0) if len(ds_val) > 0 else None
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0) if len(ds_test) > 0 else None

    if cfg.model_type == "lstm":
        model = WindLSTM(cfg).to(cfg.device)
        print(f"🧠 使用 LSTM 模型 (hidden_size={cfg.lstm_hidden_size}, num_layers={cfg.num_layers})")
    elif cfg.model_type == "hybrid":
        model = WindHybrid(cfg).to(cfg.device)
        print(f"🧠 使用 Transformer-LSTM 混合模型 (t_hidden={cfg.hidden_dim}, l_hidden={cfg.lstm_hidden_size}, fuse_dim={cfg.fuse_dim})")
    else:
        model = WindTransformer(cfg).to(cfg.device)
        print(f"🧠 使用 Transformer 模型 (hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers})")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience, min_delta=cfg.early_stopping_min_delta, mode=cfg.early_stopping_mode)

    exp_tag = f"{cfg.model_type}_{cfg.mod}_{select_fan_id}_in{cfg.input_hours}h_out{cfg.output_hours}h_{terrain_train}_{terrain_test}"

    ckpt_mod_dir = os.path.join(cfg.checkpoint_dir, exp_tag)
    os.makedirs(ckpt_mod_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_mod_dir, "best_model.pth")
    print(f"\n💾 本次实验输出目录：{ckpt_mod_dir}")

    history = {
        "train_loss": [], "train_rmse": [], "train_mae": [], "train_smape": [], "train_r2": [], "train_bias": [],
        "val_loss": [], "val_rmse": [], "val_mae": [], "val_smape": [], "val_r2": [], "val_bias": []
    }
    best_val_rmse = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        sum_tr_loss = 0.0
        tr_preds = []
        tr_tars = []
        tr_nan_samples = []

        for batch in dl_train:
            station = batch["station"].to(cfg.device)
            st_time = batch["station_time"].to(cfg.device)
            abs_time = batch["absolute_time"].to(cfg.device)
            target = batch["target"].to(cfg.device)

            pred = model(station, st_time, abs_time)
            loss = weighted_loss(pred, target, lambda_first=None)

            # 检测训练集NaN
            pred_nan = torch.isnan(pred)
            target_nan = torch.isnan(target)
            if pred_nan.any() or target_nan.any():
                batch_size = pred.size(0)
                for i in range(batch_size):
                    if pred_nan[i].any() or target_nan[i].any():
                        tr_nan_samples.append({
                            "epoch": epoch,
                            "timestamp": batch["timestamp"][i],
                            "fan_id": batch["fan_id"][i]
                        })
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_tr_loss += loss.item()
            tr_preds.append(pred.detach().cpu())
            tr_tars.append(target.cpu())

        if tr_nan_samples:
            print(f"⚠️ Epoch {epoch}: 训练集出现 {len(tr_nan_samples)} 个含NaN样本")

        avg_tr_loss = sum_tr_loss / len(dl_train) if len(dl_train) > 0 else 0.0

        if tr_preds:
            trp = torch.cat(tr_preds, dim=0).numpy()
            trt = torch.cat(tr_tars, dim=0).numpy()
            trp_real = trp * norm_stats["x_std"] + norm_stats["x_mean"]
            trt_real = trt * norm_stats["x_std"] + norm_stats["x_mean"]
            tr_metric = calculate_metrics(trp_real.flatten(), trt_real.flatten(), cfg)
        else:
            tr_metric = {'rmse': np.nan, 'mae': np.nan, 'smape': np.nan, 'r2': np.nan, 'bias': np.nan, 'n_samples': 0}

        # 验证
        model.eval()
        sum_val_loss = 0.0
        val_preds = []
        val_tars = []
        val_nan_count = 0
        val_nan_samples = []

        if dl_val is not None:
            with torch.no_grad():
                for batch in dl_val:
                    station = batch["station"].to(cfg.device)
                    st_time = batch["station_time"].to(cfg.device)
                    abs_time = batch["absolute_time"].to(cfg.device)
                    target = batch["target"].to(cfg.device)
                    pred = model(station, st_time, abs_time)

                    pred_nan = torch.isnan(pred)
                    target_nan = torch.isnan(target)
                    if pred_nan.any() or target_nan.any():
                        batch_size = pred.size(0)
                        for i in range(batch_size):
                            if pred_nan[i].any() or target_nan[i].any():
                                val_nan_samples.append({
                                    "epoch": epoch,
                                    "timestamp": batch["timestamp"][i],
                                    "fan_id": batch["fan_id"][i],
                                    "pred_has_nan": pred_nan[i].any().item(),
                                    "target_has_nan": target_nan[i].any().item()
                                })
                        val_nan_count += 1
                        continue

                    loss = weighted_loss(pred, target, lambda_first=None)
                    sum_val_loss += loss.item()
                    val_preds.append(pred.cpu())
                    val_tars.append(target.cpu())

        if val_nan_samples:
            print(f"⚠️ Epoch {epoch}: 验证集出现 {len(val_nan_samples)} 个含NaN样本（涉及 {val_nan_count} 个批次）")

        if dl_val is not None and val_preds:
            avg_val_loss = sum_val_loss / (len(dl_val) - val_nan_count) if (len(dl_val) - val_nan_count) > 0 else 0.0
            valp = torch.cat(val_preds, dim=0).numpy()
            valt = torch.cat(val_tars, dim=0).numpy()
            valp_real = valp * norm_stats["x_std"] + norm_stats["x_mean"]
            valt_real = valt * norm_stats["x_std"] + norm_stats["x_mean"]
            val_metric = calculate_metrics(valp_real.flatten(), valt_real.flatten(), cfg)
        else:
            avg_val_loss = 0.0
            val_metric = {'rmse': np.nan, 'mae': np.nan, 'smape': np.nan, 'r2': np.nan, 'bias': np.nan, 'n_samples': 0}

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_tr_loss)
        history["train_rmse"].append(tr_metric["rmse"])
        history["train_mae"].append(tr_metric["mae"])
        history["train_smape"].append(tr_metric["smape"])
        history["train_r2"].append(tr_metric["r2"])
        history["train_bias"].append(tr_metric["bias"])
        history["val_loss"].append(avg_val_loss)
        history["val_rmse"].append(val_metric["rmse"])
        history["val_mae"].append(val_metric["mae"])
        history["val_smape"].append(val_metric["smape"])
        history["val_r2"].append(val_metric["r2"])
        history["val_bias"].append(val_metric["bias"])

        print(f"\n==== Epoch {epoch} | train_loss:{avg_tr_loss:.6f} | val_loss:{avg_val_loss:.6f} ====")
        print_metrics(tr_metric, "训练集指标")
        if dl_val is not None:
            print_metrics(val_metric, "验证集指标")

        ep_ckpt = os.path.join(ckpt_mod_dir, f"epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_stats": norm_stats,
            "history": history
        }, ep_ckpt)

        if dl_val is not None and not np.isnan(val_metric["rmse"]) and val_metric["rmse"] < best_val_rmse:
            best_val_rmse = val_metric["rmse"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "norm_stats": norm_stats,
                "history": history
            }, best_ckpt_path)
            print(f"✅ 保存best_model epoch={best_epoch}, val‑RMSE={best_val_rmse:.4f}")

        if dl_val is not None:
            stop_flag = early_stop(val_metric["rmse"])
            if stop_flag:
                print(f"🛑 早停触发，best_epoch={best_epoch}")
                break

    plot_training_history(history, ckpt_mod_dir, exp_tag)

    # 加载最佳模型评估
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        norm_stats = ckpt["norm_stats"]
        ymean = norm_stats["x_mean"]
        ystd = norm_stats["x_std"]

        def eval_loader(loader):
            all_y_true = []
            all_y_pred = []
            nan_samples = []

            with torch.no_grad():
                for b in loader:
                    s = b["station"].to(cfg.device)
                    st = b["station_time"].to(cfg.device)
                    ab = b["absolute_time"].to(cfg.device)
                    tgt = b["target"].to(cfg.device)
                    pred = model(s, st, ab)

                    pred_nan = torch.isnan(pred)
                    target_nan = torch.isnan(tgt)
                    if pred_nan.any() or target_nan.any():
                        batch_size = pred.size(0)
                        for i in range(batch_size):
                            if pred_nan[i].any() or target_nan[i].any():
                                nan_samples.append({
                                    "timestamp": b["timestamp"][i],
                                    "fan_id": b["fan_id"][i],
                                    "pred_has_nan": pred_nan[i].any().item(),
                                    "target_has_nan": target_nan[i].any().item()
                                })
                        continue

                    all_y_true.append(tgt.cpu().numpy())
                    all_y_pred.append(pred.cpu().numpy())

            if all_y_true:
                yt = np.concatenate(all_y_true, axis=0)
                yp = np.concatenate(all_y_pred, axis=0)
                yt_r = yt * ystd + ymean
                yp_r = yp * ystd + ymean
                met = calculate_metrics(yt_r.flatten(), yp_r.flatten(), cfg)
                stepmet = calculate_step_metrics(yp_r, yt_r, cfg.pred_len, cfg)
            else:
                met = None
                stepmet = None

            return met, stepmet, nan_samples

        if dl_val is not None:
            val_best_met, val_step, val_nan = eval_loader(dl_val)
            print("\n" + "="*70)
            print("val‑best模型评估结果")
            print("="*70)
            print_metrics(val_best_met, "【验证集】整体指标")
            print_step_metrics(val_step, cfg.pred_len)

        if dl_test is not None:
            test_best_met, test_step, test_nan = eval_loader(dl_test)
            print_metrics(test_best_met, "【内部测试集】整体指标")
            print_step_metrics(test_step, cfg.pred_len)

            # ---- 整点时刻指标（动态） ----
            output_hours = cfg.output_hours
            # ---- 整点时刻指标：输出不含t0，0号步=10min，6号步=1h，12号=2h，18号=3h，24号=4h
            hour_idx = [0, 6, 12, 18, 24]
            hour_lab = ["10min", "1h", "2h", "3h", "4h"]
            print("\n"+"="*65)
            print(f"📊 测试集整点时刻指标 (预测{cfg.output_hours}h，不含起报t0)")
            print("="*65)
            print(f"{'时刻':>12} | {'RMSE':>10} | {'MAE':>10} | {'sMAPE':>10} | {'R²':>10} | {'Bias':>10}")
            print("-"*80)
            for idx, lab in zip(hour_idx, hour_lab):
                if idx < cfg.pred_len:
                    rmse = test_step["step_rmse"][idx]
                    mae = test_step["step_mae"][idx]
                    smape = test_step["step_smape"][idx]
                    r2 = test_step["step_r2"][idx]
                    bias = test_step["step_bias"][idx]
                    print(f"{lab:>12} | {rmse:>10.4f} | {mae:>10.4f} | {smape:>10.2f} | {r2:>10.4f} | {bias:>10.4f}")

    return history, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=str, default="cum7")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
#    parser.add_argument("--lambda_first", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--no_station_time", action="store_true")
    parser.add_argument("--no_absolute_time", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--fan_id", type=str, default="all", help="指定风机，如 F111")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm", "hybrid"],
                        help="模型类型: transformer(默认) / lstm / hybrid(Transformer-LSTM门控融合)")
    parser.add_argument("--lstm_hidden_size", type=int, default=512, help="LSTM隐藏层维度")
    parser.add_argument("--fuse_dim", type=int, default=384, help="hybrid模式融合特征维度")

    # ========== 新增参数 ==========
    parser.add_argument("--input_hours", type=int, default=4, help="输入序列覆盖的过去小时数（1~24）")
    parser.add_argument("--output_hours", type=int, default=4, help="预测的未来小时数（1~24）")
    parser.add_argument("--terrain", type=str, default="all", help="快捷设置 train/val/test 同一地形: hill(山上), low(山下), all全部")
    parser.add_argument("--terrain_train", type=str, default=None, help="训练集地形，优先级高于--terrain: hill/low/all")
    parser.add_argument("--terrain_val", type=str, default=None, help="val集地形，优先级高于--terrain: hill/low/all")
    parser.add_argument("--terrain_test", type=str, default=None, help="test集地形，优先级高于--terrain: hill/low/all")

    args = parser.parse_args()

    # ========== 打印运行命令 ==========
    import sys
    cmd = "python " + " ".join(sys.argv)
    print("="*70)
    print("运行命令:")
    print(f"  {cmd}")
    print("="*70)
    # ==================================

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = Config()
    cfg.mod = args.mod
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.hidden_dim = args.hidden_dim
    cfg.model_type = args.model_type
    cfg.lstm_hidden_size = args.lstm_hidden_size
    cfg.fuse_dim = args.fuse_dim

    cfg.num_layers = args.num_layers
    cfg.num_heads = args.num_heads
    cfg.dropout = args.dropout
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
#    cfg.lambda_first = args.lambda_first
    cfg.early_stopping_patience = args.patience
    cfg.use_station_time = not args.no_station_time
    cfg.use_absolute_time = not args.no_absolute_time
    cfg.input_hours = args.input_hours
    cfg.output_hours = args.output_hours
    cfg.seq_len = cfg.input_hours * 6 + 1
    cfg.pred_len = cfg.output_hours * 6

    if torch.cuda.is_available():
        cfg.device = torch.device("cuda")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"逻辑device {i}: {torch.cuda.get_device_name(i)}")

    print("="*70)
    print(f"Transformer风速预报模型")
    print(f"mod={cfg.mod}, input_hours={cfg.input_hours} (seq_len={cfg.seq_len}), output_hours={cfg.output_hours} (pred_len={cfg.pred_len})")
    print(f"use_station_time={cfg.use_station_time}, use_absolute_time={cfg.use_absolute_time}")
    print(f"hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}, num_heads={cfg.num_heads}")
    print(f"train:{cfg.train_start} ~ {cfg.train_end}")
    print(f"val:{cfg.val_start} ~ {cfg.val_end}")
    print(f"test:{cfg.test_start} ~ {cfg.test_end}")
    print("="*70)

    terrain_train = args.terrain_train if args.terrain_train is not None else args.terrain
    terrain_val   = args.terrain_val   if args.terrain_val   is not None else args.terrain
    terrain_test  = args.terrain_test  if args.terrain_test  is not None else args.terrain

    print(f"👉 地形设置：train={terrain_train}, val={terrain_val}, test={terrain_test}")

    train_loop(cfg,
               refresh_cache=args.refresh_cache,
               select_fan_id=args.fan_id,
               terrain_train=terrain_train,
               terrain_val=terrain_val,
               terrain_test=terrain_test)


if __name__ == "__main__":
    main()

