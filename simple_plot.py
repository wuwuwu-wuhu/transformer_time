#!/usr/bin/env python
# coding: utf-8
"""
简单的 AIS 轨迹绘制脚本
每个航线一个颜色
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(file_path, max_trajectories=30):
    """
    绘制轨迹数据，每个航线一个颜色
    先按轨迹长度排序，取前N条最长的轨迹
    
    参数:
        file_path: pkl 文件路径
        max_trajectories: 最大显示轨迹数量
    """
    # 加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"加载了 {len(data)} 条轨迹")
    
    # 按轨迹长度（点数）排序，取最长的轨迹
    data_sorted = sorted(data, key=lambda x: len(x['traj']), reverse=True)
    selected_data = data_sorted[:max_trajectories]
    
    print(f"按轨迹长度排序，选择前 {len(selected_data)} 条最长轨迹进行绘制")
    
    # 显示轨迹长度信息
    if selected_data:
        lengths = [len(traj['traj']) for traj in selected_data]
        print(f"选中轨迹长度范围: {min(lengths)} - {max(lengths)} 个点")
    
    # 创建图形
    plt.figure(figsize=(12, 8), dpi=150)
    
    # 使用不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20种颜色
    if max_trajectories > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, max_trajectories))
    
    # 绘制每条轨迹
    for i, traj_data in enumerate(selected_data):
        mmsi = traj_data['mmsi']
        traj = traj_data['traj']
        
        # 提取经纬度
        lat = traj[:, 0]  # 纬度
        lon = traj[:, 1]  # 经度
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 绘制轨迹线
        plt.plot(lon, lat, color=color, linewidth=1.5, alpha=0.8, 
                label=f'MMSI: {mmsi} ({len(traj)}点)' if i < 5 else "")
        
        # 标记起点（圆点）和终点（方块）
        plt.plot(lon[0], lat[0], 'o', color=color, markersize=5)  # 起点
        plt.plot(lon[-1], lat[-1], 's', color=color, markersize=5)  # 终点
    
    # 设置图形属性
    plt.xlabel('经度 (Longitude)', fontsize=12)
    plt.ylabel('纬度 (Latitude)', fontsize=12)
    plt.title(f'AIS 轨迹可视化 - 前{len(selected_data)}条最长航线\n○ 起点  ■ 终点', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 显示部分图例（避免过于拥挤）
    if len(selected_data) <= 5:
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # # 数据文件路径
    # data_file = r'c:\Users\zxf\Desktop\hjtra\new\TrAISformer\data\ct_dma\ct_dma_test.pkl'
    
    # # 绘制轨迹
    # plot_trajectories(data_file, max_trajectories=30)


    # # 数据文件路径
    # data_file = r'C:\Users\zxf\Desktop\hjtra\new\TrAISformer\data\ct_dma\ct_dma_train.pkl'
    
    # # 绘制轨迹
    # plot_trajectories(data_file, max_trajectories=30)

    # # 数据文件路径
    # data_file = r'C:\Users\zxf\Desktop\hjtra\new\TrAISformer\data\ct_dma\ct_dma_valid.pkl'
    
    # # 绘制轨迹
    # plot_trajectories(data_file, max_trajectories=30)

    data_file = r'C:\Users\zxf\Desktop\hjtra\data\ct_new\ct_dma_test.pkl'
    
    # 绘制轨迹
    plot_trajectories(data_file, max_trajectories=30)