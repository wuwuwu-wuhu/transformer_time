#!/usr/bin/env python
# coding: utf-8
"""
修复版本：将 AIS CSV 数据转换为 TrAISformer PKL 格式
自动检测列名，处理大文件
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import gc

def convert_csv_to_pkl(csv_path, output_path, chunk_size=100000):
    """
    转换 CSV 到 PKL 格式，支持大文件分块处理
    """
    print(f"读取 CSV 文件: {csv_path}")
    
    # 先读取一小部分检查列名
    sample_df = pd.read_csv(csv_path, nrows=10)
    print("检测到的列名:")
    for i, col in enumerate(sample_df.columns):
        print(f"  {i}: '{col}'")
    
    # 自动检测列名
    timestamp_col = None
    for col in sample_df.columns:
        if 'timestamp' in col.lower() or col.strip().lower() == 'timestamp':
            timestamp_col = col
            break
    
    if timestamp_col is None:
        # 如果没找到，使用第一列
        timestamp_col = sample_df.columns[0]
        print(f"未找到时间戳列，使用第一列: '{timestamp_col}'")
    else:
        print(f"使用时间戳列: '{timestamp_col}'")
    
    # 转换时间戳函数
    def parse_time(time_str):
        try:
            if pd.isna(time_str):
                return 0.0
            return datetime.strptime(str(time_str), "%d/%m/%Y %H:%M:%S").timestamp()
        except:
            return 0.0
    
    # 分块读取大文件
    all_trajectories = []
    total_rows = 0
    valid_rows = 0
    
    print("开始分块处理数据...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        print(f"处理第 {chunk_num + 1} 块数据 ({len(chunk)} 行)")
        total_rows += len(chunk)
        
        # 过滤有效数据
        chunk = chunk[chunk['Type of mobile'].isin(['Class A', 'Class B'])]
        chunk = chunk.dropna(subset=['MMSI', 'Latitude', 'Longitude', 'SOG', 'COG'])
        valid_rows += len(chunk)
        
        if len(chunk) == 0:
            continue
        
        # 转换时间戳
        chunk['unix_time'] = chunk[timestamp_col].apply(parse_time)
        
        # 按 MMSI 分组处理
        for mmsi, group in chunk.groupby('MMSI'):
            if len(group) < 3:  # 至少3个点
                continue
                
            # 按时间排序
            group = group.sort_values('unix_time')
            
            # 检查是否已存在该 MMSI 的轨迹
            existing_traj = None
            for i, traj in enumerate(all_trajectories):
                if traj['mmsi'] == mmsi:
                    existing_traj = i
                    break
            
            # 归一化坐标 (根据北欧海域范围)
            lat_min, lat_max = 54.0, 58.0
            lon_min, lon_max = 7.0, 13.0
            
            lat_norm = np.clip((group['Latitude'] - lat_min) / (lat_max - lat_min), 0, 0.9999)
            lon_norm = np.clip((group['Longitude'] - lon_min) / (lon_max - lon_min), 0, 0.9999)
            
            # 归一化速度和航向
            sog_norm = np.clip(group['SOG'] / 30.0, 0, 0.9999)  # 最大30节
            cog_norm = np.clip((group['COG'] % 360) / 360.0, 0, 0.9999)
            
            # 构建轨迹矩阵 (N, 6)
            new_traj = np.column_stack([
                lat_norm.values,
                lon_norm.values,
                sog_norm.values,
                cog_norm.values,
                group['unix_time'].values,
                np.full(len(group), mmsi)
            ]).astype(np.float64)
            
            if existing_traj is not None:
                # 合并到现有轨迹
                old_traj = all_trajectories[existing_traj]['traj']
                combined_traj = np.vstack([old_traj, new_traj])
                # 按时间排序
                time_order = np.argsort(combined_traj[:, 4])
                all_trajectories[existing_traj]['traj'] = combined_traj[time_order]
            else:
                # 创建新轨迹
                all_trajectories.append({
                    'mmsi': int(mmsi),
                    'traj': new_traj
                })
        
        # 清理内存
        del chunk
        gc.collect()
        
        print(f"  当前累计轨迹数: {len(all_trajectories)}")
    
    print(f"\n处理完成!")
    print(f"总行数: {total_rows}")
    print(f"有效行数: {valid_rows}")
    print(f"生成轨迹数: {len(all_trajectories)}")
    
    # 过滤太短的轨迹
    min_points = 5
    filtered_trajectories = [t for t in all_trajectories if len(t['traj']) >= min_points]
    print(f"过滤后轨迹数 (>={min_points}点): {len(filtered_trajectories)}")
    
    # 保存
    print(f"保存到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_trajectories, f)
    
    # 验证
    print("\n=== 验证结果 ===")
    if filtered_trajectories:
        t = filtered_trajectories[0]
        print(f"第一条轨迹 MMSI: {t['mmsi']}")
        print(f"轨迹形状: {t['traj'].shape}")
        print(f"前3行:")
        print(t['traj'][:3])
        
        # 统计信息
        lengths = [len(traj['traj']) for traj in filtered_trajectories]
        print(f"\n轨迹长度统计:")
        print(f"  最短: {min(lengths)} 点")
        print(f"  最长: {max(lengths)} 点")
        print(f"  平均: {np.mean(lengths):.1f} 点")
        print(f"  中位数: {np.median(lengths):.1f} 点")

if __name__ == "__main__":
    csv_file = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk-2025-02-27.csv'
    pkl_file = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk_converted.pkl'
    
    convert_csv_to_pkl(csv_file, pkl_file, chunk_size=50000)  # 减小块大小以节省内存