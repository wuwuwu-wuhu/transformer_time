#!/usr/bin/env python
# coding: utf-8
"""
绘制 AIS 轨迹数据的可视化脚本
每个航线使用不同的颜色显示
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import os

def load_trajectory_data(file_path):
    """
    加载轨迹数据
    
    参数:
        file_path: pkl 文件路径
        
    返回:
        list: 轨迹数据列表
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_all_trajectories(data, max_trajectories=50, save_path=None, figsize=(12, 10)):
    """
    绘制所有轨迹，每个航线一个颜色
    
    参数:
        data: 轨迹数据列表
        max_trajectories: 最大显示轨迹数量（避免图片过于拥挤）
        save_path: 保存路径
        figsize: 图片尺寸
    """
    plt.figure(figsize=figsize, dpi=150)
    
    # 选择要绘制的轨迹数量
    n_trajectories = min(len(data), max_trajectories)
    
    # 使用不同的颜色映射
    colors = get_cmap('tab20')  # 20种不同颜色
    if n_trajectories > 20:
        colors = get_cmap('hsv')  # 更多颜色选择
    
    print(f"绘制 {n_trajectories} 条轨迹（总共 {len(data)} 条）")
    
    for i in range(n_trajectories):
        traj_data = data[i]
        mmsi = traj_data['mmsi']
        traj = traj_data['traj']
        
        # 提取经纬度（假设列0是纬度，列1是经度）
        lat = traj[:, 0]
        lon = traj[:, 1]
        
        # 获取颜色
        color = colors(i / n_trajectories)
        
        # 绘制轨迹
        plt.plot(lon, lat, color=color, linewidth=1.5, alpha=0.8, 
                label=f'MMSI: {mmsi}' if i < 10 else "")  # 只显示前10个的标签
        
        # 标记起点和终点
        plt.plot(lon[0], lat[0], 'o', color=color, markersize=4, alpha=0.9)  # 起点
        plt.plot(lon[-1], lat[-1], 's', color=color, markersize=4, alpha=0.9)  # 终点
    
    plt.xlabel('经度 (Longitude)', fontsize=12)
    plt.ylabel('纬度 (Latitude)', fontsize=12)
    plt.title(f'AIS 轨迹可视化 - {n_trajectories} 条航线\n(圆点=起点, 方块=终点)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 只显示前10个轨迹的图例，避免过于拥挤
    if n_trajectories <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def plot_trajectories_by_region(data, max_trajectories=100, save_path=None):
    """
    按区域分组绘制轨迹
    
    参数:
        data: 轨迹数据列表
        max_trajectories: 最大显示轨迹数量
        save_path: 保存路径
    """
    # 计算所有轨迹的边界
    all_lats = []
    all_lons = []
    
    for traj_data in data[:max_trajectories]:
        traj = traj_data['traj']
        all_lats.extend(traj[:, 0])
        all_lons.extend(traj[:, 1])
    
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    
    plt.figure(figsize=(15, 10), dpi=150)
    
    # 使用颜色映射
    colors = get_cmap('viridis')
    n_trajectories = min(len(data), max_trajectories)
    
    for i in range(n_trajectories):
        traj_data = data[i]
        mmsi = traj_data['mmsi']
        traj = traj_data['traj']
        
        lat = traj[:, 0]
        lon = traj[:, 1]
        
        # 根据轨迹长度设置颜色
        color = colors(len(traj) / 1000)  # 根据轨迹点数量着色
        
        plt.plot(lon, lat, color=color, linewidth=1, alpha=0.7)
        
        # 标记起点
        plt.plot(lon[0], lat[0], 'o', color='red', markersize=2, alpha=0.8)
    
    plt.xlabel('经度 (Longitude)', fontsize=12)
    plt.ylabel('纬度 (Latitude)', fontsize=12)
    plt.title(f'AIS 轨迹区域分布 - {n_trajectories} 条航线\n(颜色深浅表示轨迹长度)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=1000))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('轨迹点数量', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"区域分布图已保存到: {save_path}")
    
    plt.show()

def plot_sample_trajectories(data, n_samples=10, save_path=None):
    """
    绘制样本轨迹的详细视图
    
    参数:
        data: 轨迹数据列表
        n_samples: 样本数量
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(min(n_samples, len(data))):
        traj_data = data[i]
        mmsi = traj_data['mmsi']
        traj = traj_data['traj']
        
        lat = traj[:, 0]
        lon = traj[:, 1]
        
        ax = axes[i]
        ax.plot(lon, lat, color=colors[i], linewidth=2, alpha=0.8)
        ax.plot(lon[0], lat[0], 'o', color=colors[i], markersize=6, label='起点')
        ax.plot(lon[-1], lat[-1], 's', color=colors[i], markersize=6, label='终点')
        
        ax.set_title(f'MMSI: {mmsi}\n({len(traj)} 个点)', fontsize=10)
        ax.set_xlabel('经度', fontsize=8)
        ax.set_ylabel('纬度', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"样本轨迹图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 数据文件路径
    data_file = r'C:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk_converted.pkl'
    
    # 输出目录
    output_dir = r'c:\Users\zxf\Desktop\hjtra\new\TrAISformer\trajectory_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载轨迹数据...")
    data = load_trajectory_data(data_file)
    print(f"成功加载 {len(data)} 条轨迹")
    
    # 1. 绘制所有轨迹概览（限制数量避免过于拥挤）
    print("\n1. 绘制轨迹概览...")
    plot_all_trajectories(
        data, 
        max_trajectories=50,
        save_path=os.path.join(output_dir, 'all_trajectories_overview.png')
    )
    
    # 2. 绘制区域分布
    print("\n2. 绘制区域分布...")
    plot_trajectories_by_region(
        data,
        max_trajectories=100,
        save_path=os.path.join(output_dir, 'trajectories_by_region.png')
    )
    
    # 3. 绘制样本轨迹详细视图
    print("\n3. 绘制样本轨迹...")
    plot_sample_trajectories(
        data,
        n_samples=10,
        save_path=os.path.join(output_dir, 'sample_trajectories.png')
    )
    
    print(f"\n所有图片已保存到: {output_dir}")

if __name__ == "__main__":
    main()