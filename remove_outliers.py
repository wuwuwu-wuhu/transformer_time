#!/usr/bin/env python
# coding: utf-8
"""
AIS 轨迹数据异常值检测和去除
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from tqdm import tqdm

def detect_speed_outliers(traj, speed_threshold=50.0):
    """
    检测速度异常值
    
    参数:
        traj: 轨迹数据 (N, 6)
        speed_threshold: 速度阈值 (节)，超过此值认为异常
        
    返回:
        mask: 布尔数组，True表示正常点
    """
    # 假设列2是归一化的SOG，需要反归一化
    sog_normalized = traj[:, 2]
    sog_actual = sog_normalized * 30.0  # 假设最大速度30节
    
    # 检测异常高速
    speed_mask = sog_actual <= speed_threshold
    
    return speed_mask

def detect_position_outliers(traj, z_threshold=3.0):
    """
    检测位置异常值（使用Z-score方法）
    
    参数:
        traj: 轨迹数据 (N, 6)
        z_threshold: Z-score阈值
        
    返回:
        mask: 布尔数组，True表示正常点
    """
    lat = traj[:, 0]
    lon = traj[:, 1]
    
    # 计算Z-score
    lat_z = np.abs(stats.zscore(lat))
    lon_z = np.abs(stats.zscore(lon))
    
    # 检测异常点
    position_mask = (lat_z < z_threshold) & (lon_z < z_threshold)
    
    return position_mask

def detect_jump_outliers(traj, max_distance=0.1):
    """
    检测位置跳跃异常值
    
    参数:
        traj: 轨迹数据 (N, 6)
        max_distance: 最大允许的相邻点距离（归一化坐标）
        
    返回:
        mask: 布尔数组，True表示正常点
    """
    if len(traj) < 2:
        return np.ones(len(traj), dtype=bool)
    
    lat = traj[:, 0]
    lon = traj[:, 1]
    
    # 计算相邻点之间的距离
    lat_diff = np.diff(lat)
    lon_diff = np.diff(lon)
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # 第一个点总是保留
    jump_mask = np.ones(len(traj), dtype=bool)
    
    # 检查跳跃
    for i in range(1, len(traj)):
        if distances[i-1] > max_distance:
            jump_mask[i] = False
    
    return jump_mask

def detect_time_outliers(traj, max_time_gap=3600):
    """
    检测时间异常值
    
    参数:
        traj: 轨迹数据 (N, 6)
        max_time_gap: 最大允许的时间间隔（秒）
        
    返回:
        mask: 布尔数组，True表示正常点
    """
    if len(traj) < 2:
        return np.ones(len(traj), dtype=bool)
    
    timestamps = traj[:, 4]
    
    # 计算时间间隔
    time_diffs = np.diff(timestamps)
    
    # 第一个点总是保留
    time_mask = np.ones(len(traj), dtype=bool)
    
    # 检查时间跳跃
    for i in range(1, len(traj)):
        if time_diffs[i-1] > max_time_gap:
            time_mask[i] = False
    
    return time_mask

def remove_outliers_from_trajectory(traj_data, config=None):
    """
    从单条轨迹中移除异常值
    
    参数:
        traj_data: 轨迹字典 {'mmsi': int, 'traj': ndarray}
        config: 配置字典
        
    返回:
        cleaned_traj_data: 清理后的轨迹数据
        outlier_info: 异常值信息
    """
    if config is None:
        config = {
            'speed_threshold': 50.0,      # 最大速度50节
            'z_threshold': 3.0,           # Z-score阈值
            'max_distance': 0.1,          # 最大跳跃距离
            'max_time_gap': 3600,         # 最大时间间隔1小时
            'min_points': 5               # 最少保留点数
        }
    
    mmsi = traj_data['mmsi']
    traj = traj_data['traj']
    original_length = len(traj)
    
    if original_length < config['min_points']:
        return None, {'reason': 'too_short', 'original_length': original_length}
    
    # 检测各种异常值
    speed_mask = detect_speed_outliers(traj, config['speed_threshold'])
    position_mask = detect_position_outliers(traj, config['z_threshold'])
    jump_mask = detect_jump_outliers(traj, config['max_distance'])
    time_mask = detect_time_outliers(traj, config['max_time_gap'])
    
    # 综合所有掩码
    combined_mask = speed_mask & position_mask & jump_mask & time_mask
    
    # 应用掩码
    cleaned_traj = traj[combined_mask]
    
    # 检查清理后的长度
    if len(cleaned_traj) < config['min_points']:
        return None, {
            'reason': 'too_few_after_cleaning',
            'original_length': original_length,
            'cleaned_length': len(cleaned_traj)
        }
    
    outlier_info = {
        'original_length': original_length,
        'cleaned_length': len(cleaned_traj),
        'removed_count': original_length - len(cleaned_traj),
        'speed_outliers': np.sum(~speed_mask),
        'position_outliers': np.sum(~position_mask),
        'jump_outliers': np.sum(~jump_mask),
        'time_outliers': np.sum(~time_mask)
    }
    
    cleaned_traj_data = {
        'mmsi': mmsi,
        'traj': cleaned_traj
    }
    
    return cleaned_traj_data, outlier_info

def clean_trajectory_dataset(input_file, output_file, config=None):
    """
    清理整个轨迹数据集
    
    参数:
        input_file: 输入PKL文件路径
        output_file: 输出PKL文件路径
        config: 配置字典
    """
    print(f"加载数据: {input_file}")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始轨迹数量: {len(data)}")
    
    cleaned_data = []
    outlier_stats = {
        'total_trajectories': len(data),
        'kept_trajectories': 0,
        'removed_trajectories': 0,
        'total_points_original': 0,
        'total_points_cleaned': 0,
        'removal_reasons': {}
    }
    
    print("开始清理轨迹...")
    
    for i, traj_data in enumerate(tqdm(data, desc="处理轨迹")):
        outlier_stats['total_points_original'] += len(traj_data['traj'])
        
        cleaned_traj, info = remove_outliers_from_trajectory(traj_data, config)
        
        if cleaned_traj is not None:
            cleaned_data.append(cleaned_traj)
            outlier_stats['kept_trajectories'] += 1
            outlier_stats['total_points_cleaned'] += len(cleaned_traj['traj'])
        else:
            outlier_stats['removed_trajectories'] += 1
            reason = info['reason']
            outlier_stats['removal_reasons'][reason] = outlier_stats['removal_reasons'].get(reason, 0) + 1
    
    # 保存清理后的数据
    print(f"保存清理后的数据: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    # 打印统计信息
    print("\n=== 清理统计 ===")
    print(f"原始轨迹数量: {outlier_stats['total_trajectories']}")
    print(f"保留轨迹数量: {outlier_stats['kept_trajectories']}")
    print(f"移除轨迹数量: {outlier_stats['removed_trajectories']}")
    print(f"保留率: {outlier_stats['kept_trajectories']/outlier_stats['total_trajectories']*100:.1f}%")
    print(f"原始总点数: {outlier_stats['total_points_original']}")
    print(f"清理后总点数: {outlier_stats['total_points_cleaned']}")
    print(f"点保留率: {outlier_stats['total_points_cleaned']/outlier_stats['total_points_original']*100:.1f}%")
    
    if outlier_stats['removal_reasons']:
        print("\n移除原因统计:")
        for reason, count in outlier_stats['removal_reasons'].items():
            print(f"  {reason}: {count} 条轨迹")
    
    return cleaned_data, outlier_stats

def visualize_cleaning_results(original_file, cleaned_file, n_samples=5):
    """
    可视化清理结果对比
    
    参数:
        original_file: 原始数据文件
        cleaned_file: 清理后数据文件
        n_samples: 显示的样本数量
    """
    # 加载数据
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)
    
    with open(cleaned_file, 'rb') as f:
        cleaned_data = pickle.load(f)
    
    # 创建对比图
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 找到在两个数据集中都存在的轨迹
    original_mmsis = {t['mmsi']: t for t in original_data}
    cleaned_mmsis = {t['mmsi']: t for t in cleaned_data}
    common_mmsis = list(set(original_mmsis.keys()) & set(cleaned_mmsis.keys()))
    
    for i in range(min(n_samples, len(common_mmsis))):
        mmsi = common_mmsis[i]
        
        orig_traj = original_mmsis[mmsi]['traj']
        clean_traj = cleaned_mmsis[mmsi]['traj']
        
        # 原始轨迹
        ax1 = axes[i, 0]
        ax1.plot(orig_traj[:, 1], orig_traj[:, 0], 'b-', alpha=0.7, linewidth=1)
        ax1.plot(orig_traj[0, 1], orig_traj[0, 0], 'go', markersize=6, label='起点')
        ax1.plot(orig_traj[-1, 1], orig_traj[-1, 0], 'ro', markersize=6, label='终点')
        ax1.set_title(f'原始轨迹 MMSI: {mmsi}\n({len(orig_traj)} 个点)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 清理后轨迹
        ax2 = axes[i, 1]
        ax2.plot(clean_traj[:, 1], clean_traj[:, 0], 'b-', alpha=0.7, linewidth=1)
        ax2.plot(clean_traj[0, 1], clean_traj[0, 0], 'go', markersize=6, label='起点')
        ax2.plot(clean_traj[-1, 1], clean_traj[-1, 0], 'ro', markersize=6, label='终点')
        ax2.set_title(f'清理后轨迹 MMSI: {mmsi}\n({len(clean_traj)} 个点)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 文件路径
    input_file = r'C:\Users\zxf\Desktop\hjtra\data\ct_new\ct_dma_test.pkl'
    output_file = r'C:\Users\zxf\Desktop\hjtra\data\ct_new\ct_dma_test_cleaned.pkl'
    
    # 配置参数
    config = {
        'speed_threshold': 40.0,      # 最大速度40节
        'z_threshold': 3.0,           # Z-score阈值
        'max_distance': 0.05,         # 最大跳跃距离（更严格）
        'max_time_gap': 7200,         # 最大时间间隔2小时
        'min_points': 10              # 最少保留点数
    }
    
    # 清理数据
    cleaned_data, stats = clean_trajectory_dataset(input_file, output_file, config)
    
    # 可视化结果
    print("\n生成对比可视化...")
    visualize_cleaning_results(input_file, output_file, n_samples=5)
    
    print(f"\n清理完成！清理后的数据保存在: {output_file}")

if __name__ == "__main__":
    main()