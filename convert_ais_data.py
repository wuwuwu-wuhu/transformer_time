#!/usr/bin/env python
# coding: utf-8
"""
简化版本：将 AIS CSV 数据转换为 TrAISformer PKL 格式
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

def convert_csv_to_pkl(csv_path, output_path):
    """
    转换 CSV 到 PKL 格式
    """
    print(f"读取 CSV 文件: {csv_path}")
    
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 行")
    
    # 过滤有效数据
    df = df[df['Type of mobile'].isin(['Class A', 'Class B'])]
    df = df.dropna(subset=['MMSI', 'Latitude', 'Longitude', 'SOG', 'COG'])
    print(f"过滤后: {len(df)} 行")
    
    # 转换时间戳
    def parse_time(time_str):
        try:
            return datetime.strptime(time_str, "%d/%m/%Y %H:%M:%S").timestamp()
        except:
            return 0.0
    
    # 注意：CSV文件中的时间戳列名是 '# Timestamp'
    timestamp_col = '# Timestamp' if '# Timestamp' in df.columns else 'Timestamp'
    df['unix_time'] = df[timestamp_col].apply(parse_time)
    
    # 按 MMSI 分组处理
    trajectories = []
    
    for mmsi, group in tqdm(df.groupby('MMSI'), desc="处理轨迹"):
        if len(group) < 5:  # 至少5个点
            continue
            
        # 按时间排序
        group = group.sort_values('unix_time')
        
        # 归一化坐标 (根据丹麦海域范围)
        lat_min, lat_max = 54.0, 58.0
        lon_min, lon_max = 7.0, 13.0
        
        lat_norm = np.clip((group['Latitude'] - lat_min) / (lat_max - lat_min), 0, 0.9999)
        lon_norm = np.clip((group['Longitude'] - lon_min) / (lon_max - lon_min), 0, 0.9999)
        
        # 归一化速度和航向
        sog_norm = np.clip(group['SOG'] / 30.0, 0, 0.9999)  # 最大30节
        cog_norm = np.clip((group['COG'] % 360) / 360.0, 0, 0.9999)
        
        # 构建轨迹矩阵 (N, 6)
        # [lat_norm, lon_norm, sog_norm, cog_norm, timestamp, mmsi]
        traj = np.column_stack([
            lat_norm.values,
            lon_norm.values,
            sog_norm.values,
            cog_norm.values,
            group['unix_time'].values,
            np.full(len(group), mmsi)
        ]).astype(np.float64)
        
        trajectories.append({
            'mmsi': int(mmsi),
            'traj': traj
        })
    
    print(f"生成 {len(trajectories)} 条轨迹")
    
    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"保存到: {output_path}")
    
    # 验证
    print("\\n=== 验证结果 ===")
    if trajectories:
        t = trajectories[0]
        print(f"第一条轨迹 MMSI: {t['mmsi']}")
        print(f"轨迹形状: {t['traj'].shape}")
        print(f"前3行:")
        print(t['traj'][:3])

if __name__ == "__main__":
    csv_file = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk-2025-02-27.csv'
    pkl_file = r'c:\Users\zxf\Desktop\hjtra\aisdk-2025-02-27\aisdk_converted.pkl'
    
    convert_csv_to_pkl(csv_file, pkl_file)