#!/usr/bin/env python
# coding: utf-8
"""
将 AIS CSV 数据转换为 TrAISformer 所需的 PKL 格式
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
from tqdm import tqdm

def parse_timestamp(timestamp_str):
    """
    解析时间戳字符串为 Unix 时间戳
    
    参数:
        timestamp_str: 时间戳字符串，格式如 "27/02/2025 00:00:00"
        
    返回:
        float: Unix 时间戳
    """
    try:
        dt = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M:%S")
        return dt.timestamp()
    except:
        return 0.0

def normalize_coordinates(lat, lon, lat_min=55.5, lat_max=58.0, lon_min=10.3, lon_max=13.0):
    """
    将经纬度归一化到 [0, 1) 区间
    
    参数:
        lat, lon: 纬度和经度
        lat_min, lat_max, lon_min, lon_max: 归一化范围
        
    返回:
        tuple: 归一化后的 (lat_norm, lon_norm)
    """
    lat_norm = (lat - lat_min) / (lat_max - lat_min)
    lon_norm = (lon - lon_min) / (lon_max - lon_min)
    
    # 确保在 [0, 1) 范围内
    lat_norm = np.clip(lat_norm, 0, 0.9999)
    lon_norm = np.clip(lon_norm, 0, 0.9999)
    
    return lat_norm, lon_norm

def normalize_sog_cog(sog, cog, sog_max=30.0):
    """
    归一化速度和航向
    
    参数:
        sog: 对地速度 (Speed Over Ground)
        cog: 对地航向 (Course Over Ground)
        sog_max: 最大速度
        
    返回:
        tuple: 归一化后的 (sog_norm, cog_norm)
    """
    # SOG 归一化到 [0, 1)
    sog_norm = np.clip(sog / sog_max, 0, 0.9999)
    
    # COG 归一化到 [0, 1)，360度对应1
    cog_norm = np.clip((cog % 360) / 360.0, 0, 0.9999)
    
    return sog_norm, cog_norm

def csv_to_pkl(csv_file_path, output_pkl_path, min_points=10):
    """
    将 CSV 文件转换为 PKL 格式
    
    参数:
        csv_file_path: 输入 CSV 文件路径
        output_pkl_path: 输出 PKL 文件路径
        min_points: 每条轨迹的最小点数
    """
    print(f"正在读取 CSV 文件: {csv_file_path}")
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    print(f"CSV 文件包含 {len(df)} 行数据")
    
    # 显示 CSV 文件的列名
    print("CSV 文件列名:", list(df.columns))
    
    # 过滤出有效的 Class A 和 Class B 数据
    valid_types = ['Class A', 'Class B']
    df_filtered = df[df['Type of mobile'].isin(valid_types)].copy()
    print(f"过滤后包含 {len(df_filtered)} 行有效数据")
    
    # 按 MMSI 分组
    grouped = df_filtered.groupby('MMSI')
    print(f"共有 {len(grouped)} 个不同的 MMSI")
    
    # 转换数据
    trajectories = []
    
    for mmsi, group in tqdm(grouped, desc="处理轨迹"):
        # 按时间排序
        group = group.sort_values('Timestamp')
        
        # 过滤掉缺失关键数据的行
        group = group.dropna(subset=['Latitude', 'Longitude', 'SOG', 'COG'])
        
        if len(group) < min_points:
            continue
            
        # 提取数据
        timestamps = group['Timestamp'].apply(parse_timestamp).values
        latitudes = group['Latitude'].values
        longitudes = group['Longitude'].values
        sogs = group['SOG'].values
        cogs = group['COG'].values
        
        # 归一化坐标
        lat_norm, lon_norm = normalize_coordinates(latitudes, longitudes)
        
        # 归一化速度和航向
        sog_norm, cog_norm = normalize_sog_cog(sogs, cogs)
        
        # 构建轨迹矩阵 (N, 6)
        # 列顺序: [lat_norm, lon_norm, sog_norm, cog_norm, timestamp, mmsi]
        traj_matrix = np.column_stack([
            lat_norm,
            lon_norm, 
            sog_norm,
            cog_norm,
            timestamps,
            np.full(len(group), mmsi)  # MMSI 列
        ])
        
        # 创建轨迹字典
        trajectory = {
            'mmsi': int(mmsi),
            'traj': traj_matrix.astype(np.float64)
        }
        
        trajectories.append(trajectory)
    
    print(f"成功处理 {len(trajectories)} 条轨迹")
    
    # 保存为 PKL 文件
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"数据已保存到: {output_pkl_path}")
    
    # 显示统计信息
    if trajectories:
        lengths = [len(t['traj']) for t in trajectories]
        print(f"轨迹长度统计:")
        print(f"  最短: {min(lengths)} 点")
        print(f"  最长: {max(lengths)} 点") 
        print(f"  平均: {np.mean(lengths):.1f} 点")
        print(f"  中位数: {np.median(lengths):.1f} 点")

def verify_conversion(pkl_file_path):
    """
    验证转换结果
    
    参数:
        pkl_file_path: PKL 文件路径
    """
    print(f"\n=== 验证转换结果: {pkl_file_path} ===")
    
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"轨迹数量: {len(data)}")
    print(f"数据类型: {type(data)}")
    
    if data:
        first_traj = data[0]
        print(f"第一个轨迹:")
        print(f"  MMSI: {first_traj['mmsi']}")
        print(f"  轨迹形状: {first_traj['traj'].shape}")
        print(f"  数据类型: {first_traj['traj'].dtype}")
        print(f"  前3行数据:")
        print(first_traj['traj'][:3])
        
        # 检查数据范围
        traj = first_traj['traj']
        print(f"  各列数据范围:")
        for i in range(traj.shape[1]):
            col_data = traj[:, i]
            print(f"    列{i}: [{col_data.min():.6f}, {col_data.max():.6f}]")

def main():
    """主函数"""
    # 文件路径
    csv_file = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk-2025-02-27.csv'
    output_dir = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27'
    output_pkl = os.path.join(output_dir, 'aisdk-2025-02-27_converted.pkl')
    
    # 检查输入文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 找不到输入文件 {csv_file}")
        return
    
    # 转换数据
    csv_to_pkl(csv_file, output_pkl, min_points=10)
    
    # 验证结果
    verify_conversion(output_pkl)

if __name__ == "__main__":
    main()