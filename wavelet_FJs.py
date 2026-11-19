import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # 小波分解库（PyWavelets，安装：pip install PyWavelets）
from datetime import datetime
import warnings
import matplotlib
from scipy.stats import pearsonr
import os
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ========== 小波分解全局参数 ==========
WAVELET_NAME = 'db8'   # 小波基（常用：db8 / sym8 / coif5）
TARGET_LEVEL = 10       # 目标分解层数（细节分量 D1~Dn 的个数），
                        # 加上近似分量 A 后共 level+1 个分量，与原 EEMD max_imf=10 的量级相当

# 1. 读取数据函数
def read_wind_data(filename):
    """读取风速数据文件（逐10分钟数据）"""
    dates = []
    wind_speeds = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()
                if len(parts) >= 3:
                    date_str = parts[0] + ' ' + parts[1]
                    wind_speed = float(parts[2])
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S'))
                    wind_speeds.append(wind_speed)

    return dates, wind_speeds

def process_fan_wavelet(fan_id, data_dir='../FJs', output_dir='./Wavelet_results',
                       start_date=None, end_date=None):
    """
    处理单个风机的小波分解（支持缺失时段检测与恢复）
    """
    print(f"\n{'='*60}")
    print(f"处理风机: {fan_id}")
    print(f"{'='*60}")

    # 读取数据
    filename = os.path.join(data_dir, f'{fan_id}.txt')
    if not os.path.exists(filename):
        print(f"警告: 文件 {filename} 不存在")
        return None

    dates, wind_speeds = read_wind_data(filename)
    print(f"读取数据: {len(wind_speeds)} 条记录")
    print(f"时间范围: {dates[0]} 到 {dates[-1]}")

    # 如果指定了日期范围，进行筛选
    if start_date is not None or end_date is not None:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else dates[0]
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else dates[-1]
        filtered_dates = []
        filtered_wind = []
        for i, date in enumerate(dates):
            if date >= start_dt and date <= end_dt:
                filtered_dates.append(date)
                filtered_wind.append(wind_speeds[i])
        dates = filtered_dates
        wind_speeds = filtered_wind
        print(f"筛选后数据: {len(wind_speeds)} 条记录")
        print(f"筛选后时间范围: {dates[0]} 到 {dates[-1]}")

    if len(wind_speeds) == 0:
        print("警告：没有数据！")
        return None

    # 创建输出目录
    fan_output_dir = os.path.join(fan_id)
    os.makedirs(fan_output_dir, exist_ok=True)

    # ========== 数据连续性检查与重采样 ==========
    print("\n=== 数据连续性检查与重采样 ===")
    # 创建DataFrame
    df_orig = pd.DataFrame({'speed': wind_speeds}, index=dates)
    df_orig = df_orig.sort_index()

    # 创建完整10分钟时间索引
    start_dt = df_orig.index.min()
    end_dt = df_orig.index.max()
    full_index = pd.date_range(start=start_dt, end=end_dt, freq='10min')
    print(f"完整10分钟网格: {len(full_index)} 个点")

    # 重采样到完整网格，缺失处为NaN
    df_full = df_orig.reindex(full_index)

    # 检测缺失段
    missing_mask = df_full['speed'].isna()
    if missing_mask.any():
        # 统计缺失段
        missing_indices = np.where(missing_mask)[0]
        missing_segments = []
        start_idx = missing_indices[0]
        for i in range(1, len(missing_indices)):
            if missing_indices[i] != missing_indices[i-1] + 1:
                missing_segments.append((full_index[start_idx], full_index[missing_indices[i-1]]))
                start_idx = missing_indices[i]
        missing_segments.append((full_index[start_idx], full_index[missing_indices[-1]]))

        total_missing = missing_mask.sum()
        print(f"⚠️ 发现 {len(missing_segments)} 个缺失段，共 {total_missing} 个时间点缺失")
        for i, (s, e) in enumerate(missing_segments[:5]):
            print(f"  {i+1}. {s} ~ {e}")
        if len(missing_segments) > 5:
            print(f"  ... 等 {len(missing_segments)-5} 个缺失段")
    else:
        print("✅ 数据完整，无缺失")
        missing_mask = None

    # 线性插值填补缺失值，得到连续序列用于小波分解
    df_interp = df_full.interpolate(method='linear', limit_direction='both')
    signal = df_interp['speed'].values
    signal_dates = df_interp.index.to_pydatetime()
    print(f"插值后序列长度: {len(signal)}")
    # ============================================

    # 3. 进行小波分解（使用插值后的连续序列）
    print("\n正在进行小波分解...")
    signal = signal.flatten()

    # 根据信号长度与小波滤波器长度，确定最大可行分解层数
    wavelet_obj = pywt.Wavelet(WAVELET_NAME)
    max_level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)
    level = min(TARGET_LEVEL, max_level)
    print(f"小波基: {WAVELET_NAME}, 分解层数: {level}"
          f"（D1~D{level} 细节 + A 近似，共 {level+1} 个分量）")

    try:
        # wavedec 返回 [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        #   coeffs[0]   = 近似系数 cA_n（最低频/趋势）
        #   coeffs[1]   = cD_n（最粗细节）
        #   ...
        #   coeffs[n]   = cD_1（最细细节，最高频）
        coeffs = pywt.wavedec(signal, WAVELET_NAME, level=level, mode='symmetric')
        n_levels = len(coeffs) - 1  # 细节分量个数 = level

        # 用 waverec 把每个尺度单独重构出来（其他系数置零而非None，
        # 以保证边界处理一致、各分量求和可完美还原原信号），
        # 并按"高频在前、低频/趋势在后"排列：
        #   IMFs[0] = D1（高频）, ..., IMFs[n_levels-1] = Dn, IMFs[n_levels] = A（趋势）
        IMFs = []
        for j in range(1, n_levels + 1):
            idx = n_levels - j + 1      # cD_j 在 coeffs 中的位置
            c = [np.zeros_like(x) for x in coeffs]
            c[idx] = coeffs[idx]
            rec = pywt.waverec(c, WAVELET_NAME, mode='symmetric')
            IMFs.append(rec[:len(signal)])
        # 近似分量 A
        c = [np.zeros_like(x) for x in coeffs]
        c[0] = coeffs[0]
        rec = pywt.waverec(c, WAVELET_NAME, mode='symmetric')
        IMFs.append(rec[:len(signal)])

        IMFs = np.array(IMFs)
        n_imfs = IMFs.shape[0]
        # 分量名称：D1, D2, ..., Dn, A
        comp_names = [f'D{j}' for j in range(1, n_levels + 1)] + ['A']
        print(f"分解得到 {n_imfs} 个小波分量（D1~D{n_levels} 细节 + A 近似）")
    except Exception as e:
        print(f"小波分解出错: {e}")
        raise

    # 4. 相关系数计算（与原来相同，使用插值后的信号）
    def calculate_correlation(original, component_sum):
        correlation, p_value = pearsonr(original, component_sum)
        return correlation, p_value

    # 5. 绘图1：原始风速和累积和分量（带相关系数）
    max_plots = min(n_imfs, 12)  # 最多显示12个累积和
    fig1, axes1 = plt.subplots(max_plots + 1, 1, figsize=(15, 3*(max_plots+1)), sharex=True)

    # 存储相关系数
    correlations = []

    # 原始风速
    axes1[0].plot(signal_dates, signal, 'k-', linewidth=1.5, label='原始风速')
    axes1[0].set_ylabel('风速 (m/s)')
    axes1[0].set_title('原始风速序列')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # 绘制累积和分量（后1~i个分量的和，即越来越多的低频部分）
    for i in range(1, max_plots + 1):
        # 计算后i个分量的和
        cumulative_sum = np.sum(IMFs[-i:], axis=0)

        # 计算相关系数
        corr, p_value = calculate_correlation(signal, cumulative_sum)
        correlations.append((i, corr, p_value))

        range_label = f"{comp_names[n_imfs-i]}到{comp_names[n_imfs-1]}"
        axes1[i].plot(signal_dates, signal, 'k-', linewidth=1.0, alpha=0.5, label='原始风速')
        axes1[i].plot(signal_dates, cumulative_sum, 'r-', linewidth=1.5,
                     label=f'后{i}个分量之和 (r={corr:.4f})')
        axes1[i].set_ylabel('风速 (m/s)')
        axes1[i].set_title(f'后{i}个小波分量的和 ({range_label})')
        axes1[i].legend(loc='upper right')
        axes1[i].grid(True, alpha=0.3)

        axes1[i].text(0.02, 0.95, f'相关系数: {corr:.4f}\np值: {p_value:.2e}',
                     transform=axes1[i].transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes1[-1].set_xlabel('时间')
    plt.suptitle(f'风机{fan_id}: 原始风速与后i个小波分量的和（含相关系数）', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fan_output_dir, f'{fan_id}_wavelet_cumulative_components.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {fan_output_dir}/{fan_id}_wavelet_cumulative_components.pdf")

    # 6. 绘图2：原始风速和各小波分量
    max_plots_imf = min(n_imfs, 12)
    fig2, axes2 = plt.subplots(max_plots_imf + 1, 1, figsize=(15, 3*(max_plots_imf+1)), sharex=True)

    imf_correlations = []

    # 原始风速
    axes2[0].plot(signal_dates, signal, 'k-', linewidth=1.5, label='原始风速')
    axes2[0].set_ylabel('风速 (m/s)')
    axes2[0].set_title('原始风速序列')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    # 绘制各个小波分量
    for i in range(max_plots_imf):
        corr, p_value = calculate_correlation(signal, IMFs[i])
        imf_correlations.append((i+1, corr, p_value))

        axes2[i+1].plot(signal_dates, signal, 'k-', linewidth=1.0, alpha=0.3, label='原始风速')
        axes2[i+1].plot(signal_dates, IMFs[i], 'b-', linewidth=1.5,
                       label=f'{comp_names[i]} (r={corr:.4f})')
        axes2[i+1].set_ylabel('幅值')
        axes2[i+1].set_title(f'第{i+1}个小波分量 ({comp_names[i]})')
        axes2[i+1].legend(loc='upper right')
        axes2[i+1].grid(True, alpha=0.3)

        axes2[i+1].text(0.02, 0.95, f'相关系数: {corr:.4f}\np值: {p_value:.2e}',
                       transform=axes2[i+1].transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    axes2[-1].set_xlabel('时间')
    plt.suptitle(f'风机{fan_id}: 原始风速与各小波分量（含相关系数）', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fan_output_dir, f'{fan_id}_wavelet_individual_components.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {fan_output_dir}/{fan_id}_wavelet_individual_components.pdf")

    # 7. 输出详细的相关系数统计
    print("\n=== 后i个小波分量的和与原始序列的相关系数 ===")
    print(f"{'后i个分量':<12} {'分量范围':<20} {'相关系数':<12} {'p值':<12} {'显著性'}")
    print("-" * 70)

    for i, corr, p_value in correlations:
        comp_range = f"{comp_names[n_imfs-i]}-{comp_names[n_imfs-1]}"
        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        print(f"后{i}个分量{' ':>4} {comp_range:<20} {corr:>10.6f}  {p_value:>10.2e}  {significance}")

    print("\n显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")

    print("\n=== 各小波分量与原始序列的相关系数 ===")
    print(f"{'分量':<10} {'相关系数':<12} {'p值':<12} {'显著性'}")
    print("-" * 50)

    for i, corr, p_value in imf_correlations:
        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        print(f"{comp_names[i-1]:<8}   {corr:>10.6f}  {p_value:>10.2e}  {significance}")

    # 8. 找出最佳累积分量
    if correlations:
        best_i, best_corr, best_p = max(correlations, key=lambda x: abs(x[1]))
        best_range = f"{comp_names[n_imfs-best_i]}-{comp_names[n_imfs-1]}"
        print(f"\n最佳累积分量: 后{best_i}个分量 ({best_range})")
        print(f"相关系数: {best_corr:.6f}, p值: {best_p:.2e}")

    # 9. 输出统计信息
    print("\n=== 小波分解统计信息 ===")
    print(f"总数据点数: {len(signal)}")
    print(f"分解得到的小波分量数: {n_imfs}")

    total_variance = np.var(signal)
    print(f"\n各分量方差贡献率:")
    for i in range(n_imfs):
        component_variance = np.var(IMFs[i])
        variance_ratio = (component_variance / total_variance) * 100
        print(f"{comp_names[i]}: {variance_ratio:.2f}%")

    reconstructed = np.sum(IMFs, axis=0)
    reconstruction_error = np.mean((signal - reconstructed) ** 2)
    print(f"\n重构误差(MSE): {reconstruction_error:.6f}")

    # 10. 绘制相关系数变化图
    if correlations:
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        i_values = [c[0] for c in correlations]
        corr_values = [c[1] for c in correlations]

        total_variance = np.var(signal)
        cumulative_variance_ratios = []

        for i in range(1, n_imfs + 1):
            cumulative_var = np.sum([np.var(IMFs[-j]) for j in range(1, i+1)])
            cumulative_ratio = (cumulative_var / total_variance) * 100
            cumulative_variance_ratios.append(cumulative_ratio)

        bars = ax3.bar(range(len(i_values)), corr_values, color='steelblue', alpha=0.7)
        ax3.set_xlabel('后i个分量的和', fontsize=12)
        ax3.set_ylabel('相关系数', fontsize=12)
        ax3.set_title(f'风机{fan_id}: 后i个小波分量的和与原始风速序列的相关系数', fontsize=14)
        ax3.set_xticks(range(len(i_values)))
        ax3.set_xticklabels([f'后{i}个' for i in i_values])
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, corr in zip(bars, corr_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{corr:.4f}', ha='center', va='bottom', fontsize=9)

        for bar, cum_var in zip(bars, cumulative_variance_ratios[:len(correlations)]):
            bar_height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., bar_height/2,
                     f'累积方差\n{cum_var:.1f}%', ha='center', va='center',
                     fontsize=8, color='red', fontweight='bold')

        ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='r=0.9')
        ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='r=0.7')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fan_output_dir, f'{fan_id}_correlation_analysis.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {fan_output_dir}/{fan_id}_correlation_analysis.pdf")

    # 11. 保存序列到文件
    print("\n=== 保存序列数据（缺失时段恢复为NaN） ===")

    # 计算累积和（使用小波分量，这些是连续值）
    cumulative_sums = []
    for i in range(1, n_imfs + 1):
        cum_sum = np.sum(IMFs[-i:], axis=0)
        cumulative_sums.append(cum_sum)

    # 分组（根据分量数动态调整，高频噪声 -> 低频趋势）
    if n_imfs >= 11:
        high_freq_noise = np.sum(IMFs[:2], axis=0)
        mid_high_freq = np.sum(IMFs[2:5], axis=0)
        mid_low_freq = np.sum(IMFs[5:8], axis=0)
        low_freq_trend = np.sum(IMFs[8:], axis=0)
        group_names = ['高频(D1-D2)', '中高频(D3-D5)', '中低频(D6-D8)', '低频(D9-A)']
        groups = [high_freq_noise, mid_high_freq, mid_low_freq, low_freq_trend]
        n_groups = 4
    elif n_imfs >= 8:
        high_freq_noise = np.sum(IMFs[:2], axis=0)
        mid_freq = np.sum(IMFs[2:5], axis=0)
        low_freq_trend = np.sum(IMFs[5:], axis=0)
        group_names = ['高频(D1-D2)', '中频(D3-D5)', '低频(D6-A)']
        groups = [high_freq_noise, mid_freq, low_freq_trend]
        n_groups = 3
    else:
        group_names = []
        groups = []
        n_groups = 0

    # 构建列名
    column_names = ['时间', '原始序列']
    column_names += comp_names
    column_names += [f'后{i}个累加' for i in range(1, n_imfs + 1)]
    if n_groups > 0:
        column_names += group_names

    num_rows = len(signal_dates)  # 插值后的连续序列长度（= full_index 长度）
    num_cols = 1 + 1 + n_imfs + n_imfs + n_groups

    # 创建输出数据数组，全部填 NaN
    data = np.full((num_rows, num_cols), np.nan, dtype=np.float64)
    time_strs = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in signal_dates]

    # 原始序列：使用重采样后的原始数据（含NaN）
    data[:, 0] = df_full['speed'].values   # 原始序列（NaN保留）

    # 小波分量、累积和、分组：这些来自小波分解，是连续值
    for i in range(n_imfs):
        data[:, 1 + i] = IMFs[i]
    for i in range(n_imfs):
        data[:, 1 + n_imfs + i] = cumulative_sums[i]
    for i in range(n_groups):
        data[:, 1 + n_imfs + n_imfs + i] = groups[i]

    # 将缺失位置全部置为 NaN（不仅仅是原始序列，而是所有列）
    if missing_mask is not None:
        # 注意：missing_mask 与 full_index 对齐
        data[missing_mask, :] = np.nan

    # 保存文件
    output_file = os.path.join(fan_output_dir, f'{fan_id}_wavelet_decomposition.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\t'.join(column_names) + '\n')
        for row_idx in range(num_rows):
            row_data = [time_strs[row_idx]]
            for col_idx in range(num_cols):
                val = data[row_idx, col_idx]
                if np.isnan(val):
                    row_data.append('')   # 输出空字符串表示缺失
                else:
                    row_data.append(f'{val:.6f}')
            f.write('\t'.join(row_data) + '\n')

    print(f"数据已保存到: {output_file}")
    print(f"文件包含: {num_rows} 行数据, {num_cols + 1} 列（含时间列）")
    if missing_mask is not None:
        print(f"其中缺失时段已标记为NaN（空值）")

    # 返回结果
    return {
        'fan_id': fan_id,
        'n_imfs': n_imfs,
        'signal': signal,               # 插值后的连续信号
        'IMFs': IMFs,
        'comp_names': comp_names,
        'wavelet': WAVELET_NAME,
        'level': level,
        'dates': signal_dates,
        'correlations': correlations,
        'imf_correlations': imf_correlations
    }


def process_all_fans(fan_list=None, data_dir='../FJs', output_dir='./Wavelet_results',
                     start_date=None, end_date=None):
    """
    处理所有风机

    参数:
    fan_list: 风机ID列表，如果为None则自动查找所有F???.txt文件
    data_dir: 数据文件目录
    output_dir: 输出目录
    start_date: 起始日期
    end_date: 结束日期
    """
    print("="*60)
    print("小波分解分析程序")
    print("="*60)

    if fan_list is None:
        import glob
        files = glob.glob(os.path.join(data_dir, 'F???.txt'))
        fan_list = [os.path.basename(f).replace('.txt', '') for f in files]
        fan_list = sorted(fan_list)

    print(f"发现 {len(fan_list)} 个风机")

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for fan_id in fan_list:
        result = process_fan_wavelet(fan_id, data_dir, output_dir, start_date, end_date)
        if result:
            results[fan_id] = result

    print("\n" + "="*60)
    print("所有风机处理完成!")
    print(f"成功处理 {len(results)} 个风机")
    print(f"结果保存在: {output_dir}")
    print("="*60)

    return results


if __name__ == "__main__":
    # 示例：处理单个风机
#    process_fan_wavelet('F111', data_dir='../FJs', output_dir='./Wavelet_results')

    # 示例：处理所有风机
    # process_all_fans(data_dir='../FJs', output_dir='./Wavelet_results')

    # 示例：指定日期范围
    # process_fan_wavelet('F111', data_dir='../FJs', output_dir='./Wavelet_results',
    #                  start_date='2023-07-01', end_date='2023-09-01')

    # 示例：处理指定风机列表
    # fan_list = ['F111', 'F112', 'F113']
    # process_all_fans(fan_list=fan_list, data_dir='../FJs', output_dir='./Wavelet_results')

    # 默认：处理所有风机
    process_all_fans(data_dir='../FJs', output_dir='./Wavelet_results')

