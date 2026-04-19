#!/usr/bin/env python
# coding: utf-8
"""
绘制清洗后数据中 MMSI 轨迹数据量排行前30的图
按轨迹条数（不是长度）排序
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_mmsi_counts(pkl_file):
    """
    加载数据并分析每个 MMSI 的轨迹数量
    
    参数:
        pkl_file: PKL 文件路径
        
    返回:
        mmsi_counts: Counter 对象，包含每个 MMSI 的轨迹数量
        data: 原始数据
    """
    print(f"加载数据: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"总轨迹数量: {len(data)}")
    
    # 统计每个 MMSI 的轨迹数量
    mmsi_list = [traj['mmsi'] for traj in data]
    mmsi_counts = Counter(mmsi_list)
    
    print(f"不同 MMSI 数量: {len(mmsi_counts)}")
    
    return mmsi_counts, data

def plot_top_mmsi_counts(mmsi_counts, top_n=30, save_path=None):
    """
    绘制 MMSI 轨迹数量排行榜
    
    参数:
        mmsi_counts: Counter 对象
        top_n: 显示前N个
        save_path: 保存路径
    """
    # 获取前N个 MMSI
    top_mmsis = mmsi_counts.most_common(top_n)
    
    mmsis = [str(mmsi) for mmsi, count in top_mmsis]
    counts = [count for mmsi, count in top_mmsis]
    
    # 创建柱状图
    plt.figure(figsize=(15, 8), dpi=150)
    
    # 使用不同颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(mmsis)))
    
    bars = plt.bar(range(len(mmsis)), counts, color=colors, alpha=0.8)
    
    # 设置标签和标题
    plt.xlabel('MMSI', fontsize=12)
    plt.ylabel('轨迹数量', fontsize=12)
    plt.title(f'前{top_n}个 MMSI 轨迹数量排行榜', fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    plt.xticks(range(len(mmsis)), mmsis, rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 前{top_n}个 MMSI 统计 ===")
    for i, (mmsi, count) in enumerate(top_mmsis):
        print(f"{i+1:2d}. MMSI: {mmsi} - {count} 条轨迹")

def plot_individual_trajectories(data, mmsi_counts, top_n=30, max_plots=6):
    """
    分别绘制前N个 MMSI 的轨迹图
    
    参数:
        data: 轨迹数据
        mmsi_counts: MMSI 计数
        top_n: 前N个 MMSI
        max_plots: 最多绘制几个 MMSI 的轨迹
    """
    # 获取前N个 MMSI
    top_mmsis = mmsi_counts.most_common(top_n)
    
    # 按 MMSI 分组数据
    mmsi_data = {}
    for traj in data:
        mmsi = traj['mmsi']
        if mmsi not in mmsi_data:
            mmsi_data[mmsi] = []
        mmsi_data[mmsi].append(traj)
    
    # 绘制前几个 MMSI 的轨迹
    plot_count = min(max_plots, len(top_mmsis))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i in range(plot_count):
        mmsi, count = top_mmsis[i]
        trajectories = mmsi_data[mmsi]
        
        ax = axes[i]
        
        # 绘制该 MMSI 的所有轨迹
        for j, traj_data in enumerate(trajectories):
            traj = traj_data['traj']
            lat = traj[:, 0]
            lon = traj[:, 1]
            
            # 使用不同的线型区分不同轨迹
            linestyle = '-' if j < 5 else '--' if j < 10 else ':'
            alpha = 0.8 if j < 5 else 0.6 if j < 10 else 0.4
            
            ax.plot(lon, lat, color=colors[i], linewidth=1.5, 
                   alpha=alpha, linestyle=linestyle,
                   label=f'轨迹 {j+1}' if j < 3 else "")
            
            # 标记起点和终点
            ax.plot(lon[0], lat[0], 'o', color=colors[i], markersize=4, alpha=0.9)
            ax.plot(lon[-1], lat[-1], 's', color=colors[i], markersize=4, alpha=0.9)
        
        ax.set_title(f'MMSI: {mmsi}\n({count} 条轨迹)', fontsize=12, fontweight='bold')
        ax.set_xlabel('经度 (归一化)', fontsize=10)
        ax.set_ylabel('纬度 (归一化)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 只为前几个显示图例
        if i < 3 and count <= 5:
            ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(plot_count, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'前{plot_count}个 MMSI 的轨迹分布图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_mmsi_distribution(mmsi_counts, save_path=None):
    """
    绘制 MMSI 轨迹数量分布直方图
    
    参数:
        mmsi_counts: MMSI 计数
        save_path: 保存路径
    """
    counts = list(mmsi_counts.values())
    
    plt.figure(figsize=(12, 6), dpi=150)
    
    # 绘制直方图
    plt.hist(counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.xlabel('每个 MMSI 的轨迹数量', fontsize=12)
    plt.ylabel('MMSI 数量', fontsize=12)
    plt.title('MMSI 轨迹数量分布直方图', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    max_count = max(counts)
    
    plt.axvline(mean_count, color='red', linestyle='--', alpha=0.8, label=f'平均值: {mean_count:.1f}')
    plt.axvline(median_count, color='green', linestyle='--', alpha=0.8, label=f'中位数: {median_count:.1f}')
    
    plt.legend()
    
    # 显示统计信息
    plt.text(0.7, 0.8, f'总 MMSI 数: {len(mmsi_counts)}\n'
                       f'最大轨迹数: {max_count}\n'
                       f'平均轨迹数: {mean_count:.1f}\n'
                       f'中位数轨迹数: {median_count:.1f}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"分布图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 文件路径
    cleaned_file = r'C:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk_converted.pkl'
    output_dir = r'C:\Users\zxf\Desktop\hjtra\new\TrAISformer\mmsi_analysis'
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载和分析数据
    mmsi_counts, data = load_and_analyze_mmsi_counts(cleaned_file)
    
    # 1. 绘制 MMSI 轨迹数量排行榜
    print("\n1. 绘制 MMSI 轨迹数量排行榜...")
    plot_top_mmsi_counts(
        mmsi_counts, 
        top_n=30,
        save_path=os.path.join(output_dir, 'top30_mmsi_counts.png')
    )
    
    # 2. 绘制 MMSI 轨迹数量分布
    print("\n2. 绘制 MMSI 轨迹数量分布...")
    plot_mmsi_distribution(
        mmsi_counts,
        save_path=os.path.join(output_dir, 'mmsi_distribution.png')
    )
    
    # 3. 分别绘制前6个 MMSI 的轨迹
    print("\n3. 绘制前6个 MMSI 的轨迹分布...")
    plot_individual_trajectories(data, mmsi_counts, top_n=30, max_plots=6)
    
    print(f"\n所有图片已保存到: {output_dir}")

if __name__ == "__main__":
    main()