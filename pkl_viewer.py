# coding=utf-8
"""
PKL 文件查看器
用于快速查看和分析 pickle 文件内容
"""

import pickle
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import argparse


def analyze_data_structure(data, name="data"):
    """
    分析数据结构
    
    参数:
        data: 要分析的数据
        name: 数据名称
        
    返回:
        dict: 分析结果
    """
    analysis = {
        'name': name,
        'type': str(type(data)),
        'size': None,
        'shape': None,
        'keys': None,
        'sample': None
    }
    
    # 获取大小信息
    if hasattr(data, '__len__'):
        analysis['size'] = len(data)
    
    # 获取形状信息
    if hasattr(data, 'shape'):
        analysis['shape'] = data.shape
    
    # 获取键信息（字典类型）
    if isinstance(data, dict):
        analysis['keys'] = list(data.keys())
        if len(data) > 0:
            first_key = list(data.keys())[0]
            analysis['sample'] = {first_key: str(data[first_key])[:200]}
    
    # 获取样本数据
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        analysis['sample'] = str(data[0])[:200]
    elif isinstance(data, np.ndarray) and data.size > 0:
        analysis['sample'] = str(data.flat[:5])[:200]
    elif not isinstance(data, (dict, list, tuple, np.ndarray)):
        analysis['sample'] = str(data)[:200]
    
    return analysis


def view_trajectory_data(data):
    """
    专门查看轨迹数据
    
    参数:
        data: 轨迹数据列表
    """
    print("\n" + "="*60)
    print("轨迹数据详细分析")
    print("="*60)
    
    if not isinstance(data, list):
        print("❌ 数据不是轨迹格式（应该是列表）")
        return
    
    print(f"📊 轨迹总数: {len(data)}")
    
    if len(data) == 0:
        print("❌ 没有轨迹数据")
        return
    
    # 分析第一条轨迹
    first_traj = data[0]
    print(f"\n🔍 第一条轨迹分析:")
    print(f"   类型: {type(first_traj)}")
    
    if isinstance(first_traj, dict):
        print(f"   键: {list(first_traj.keys())}")
        
        if 'mmsi' in first_traj:
            print(f"   MMSI: {first_traj['mmsi']}")
        
        if 'traj' in first_traj:
            traj = first_traj['traj']
            print(f"   轨迹形状: {traj.shape}")
            print(f"   数据类型: {traj.dtype}")
            
            if len(traj) > 0:
                print(f"   前3个点:")
                for i, point in enumerate(traj[:3]):
                    print(f"     点{i+1}: {point}")
                
                # 分析数据范围
                print(f"\n📈 数据范围分析:")
                if traj.shape[1] >= 4:
                    print(f"   纬度: {traj[:, 0].min():.6f} - {traj[:, 0].max():.6f}")
                    print(f"   经度: {traj[:, 1].min():.6f} - {traj[:, 1].max():.6f}")
                    print(f"   速度: {traj[:, 2].min():.6f} - {traj[:, 2].max():.6f}")
                    print(f"   航向: {traj[:, 3].min():.6f} - {traj[:, 3].max():.6f}")
    
    # 统计轨迹长度
    traj_lengths = []
    mmsi_list = []
    
    for i, traj_data in enumerate(data[:100]):  # 只分析前100条
        if isinstance(traj_data, dict) and 'traj' in traj_data:
            traj_lengths.append(len(traj_data['traj']))
            if 'mmsi' in traj_data:
                mmsi_list.append(traj_data['mmsi'])
    
    if traj_lengths:
        print(f"\n📏 轨迹长度统计 (前100条):")
        print(f"   平均长度: {np.mean(traj_lengths):.1f}")
        print(f"   最短轨迹: {min(traj_lengths)}")
        print(f"   最长轨迹: {max(traj_lengths)}")
        print(f"   中位数: {np.median(traj_lengths):.1f}")
    
    if mmsi_list:
        unique_mmsi = len(set(mmsi_list))
        print(f"\n🚢 MMSI 统计 (前100条):")
        print(f"   唯一船舶数: {unique_mmsi}")
        print(f"   示例MMSI: {mmsi_list[:5]}")


def view_coastline_data(data):
    """
    专门查看海岸线数据
    
    参数:
        data: 海岸线数据
    """
    print("\n" + "="*60)
    print("海岸线数据详细分析")
    print("="*60)
    
    if isinstance(data, list):
        print(f"📊 多边形数量: {len(data)}")
        
        total_points = 0
        for i, polygon in enumerate(data):
            if hasattr(polygon, 'shape'):
                points = polygon.shape[0]
                total_points += points
                print(f"   多边形 {i+1}: {points} 个点")
                
                if i == 0:  # 显示第一个多边形的详细信息
                    print(f"   第一个多边形坐标范围:")
                    print(f"     经度: {polygon[:, 0].min():.4f} - {polygon[:, 0].max():.4f}")
                    print(f"     纬度: {polygon[:, 1].min():.4f} - {polygon[:, 1].max():.4f}")
        
        print(f"\n📈 总计: {total_points} 个坐标点")


def view_pkl_file(file_path, detailed=False):
    """
    查看 PKL 文件内容
    
    参数:
        file_path: PKL 文件路径
        detailed: 是否显示详细信息
    """
    print(f"\n🔍 正在分析文件: {file_path}")
    print("-" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        # 加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 基本分析
        analysis = analyze_data_structure(data, os.path.basename(file_path))
        
        print(f"📁 文件名: {analysis['name']}")
        print(f"📊 数据类型: {analysis['type']}")
        print(f"📏 数据大小: {analysis['size']}")
        print(f"🔢 数据形状: {analysis['shape']}")
        
        if analysis['keys']:
            print(f"🔑 字典键: {analysis['keys']}")
        
        if analysis['sample']:
            print(f"📝 样本数据: {analysis['sample']}")
        
        # 详细分析
        if detailed:
            filename = os.path.basename(file_path).lower()
            
            if 'coastline' in filename:
                view_coastline_data(data)
            elif any(x in filename for x in ['train', 'valid', 'test', 'traj']):
                view_trajectory_data(data)
            else:
                print(f"\n💡 提示: 使用 --detailed 参数查看更多信息")
        
        # 文件信息
        file_size = os.path.getsize(file_path)
        print(f"\n💾 文件大小: {file_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PKL 文件查看器')
    parser.add_argument('file_path', nargs='?', help='PKL 文件路径')
    parser.add_argument('--detailed', '-d', action='store_true', help='显示详细信息')
    parser.add_argument('--all', '-a', action='store_true', help='查看目录中所有 PKL 文件')
    
    args = parser.parse_args()
    
    print("🔍 PKL 文件查看器")
    print("=" * 60)
    
    if args.all:
        # 查看当前目录和 data 目录中的所有 PKL 文件
        search_dirs = ['.', 'data', 'data/ct_dma']
        pkl_files = []
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.pkl'):
                        pkl_files.append(os.path.join(search_dir, file))
        
        if pkl_files:
            print(f"📁 找到 {len(pkl_files)} 个 PKL 文件:")
            for i, file_path in enumerate(pkl_files, 1):
                print(f"{i}. {file_path}")
            
            print("\n" + "="*60)
            for file_path in pkl_files:
                view_pkl_file(file_path, args.detailed)
        else:
            print("❌ 没有找到 PKL 文件")
    
    elif args.file_path:
        view_pkl_file(args.file_path, args.detailed)
    
    else:
        # 交互式选择
        data_dir = 'data/ct_dma'
        if os.path.exists(data_dir):
            pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            
            if pkl_files:
                print(f"📁 在 {data_dir} 中找到以下 PKL 文件:")
                for i, filename in enumerate(pkl_files, 1):
                    file_path = os.path.join(data_dir, filename)
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    print(f"{i}. {filename} ({file_size:.1f} MB)")
                
                try:
                    choice = input(f"\n请选择要查看的文件 (1-{len(pkl_files)}) 或按 Enter 查看全部: ")
                    
                    if choice.strip() == "":
                        # 查看全部
                        for filename in pkl_files:
                            file_path = os.path.join(data_dir, filename)
                            view_pkl_file(file_path, True)
                    else:
                        idx = int(choice) - 1
                        if 0 <= idx < len(pkl_files):
                            file_path = os.path.join(data_dir, pkl_files[idx])
                            view_pkl_file(file_path, True)
                        else:
                            print("❌ 无效选择")
                
                except (ValueError, KeyboardInterrupt):
                    print("\n👋 退出查看器")
            else:
                print(f"❌ 在 {data_dir} 中没有找到 PKL 文件")
        else:
            print("❌ 数据目录不存在，请指定 PKL 文件路径")
            print("用法: python pkl_viewer.py <文件路径>")


if __name__ == "__main__":
    main()
