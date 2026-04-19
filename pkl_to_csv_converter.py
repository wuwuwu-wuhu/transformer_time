# coding=utf-8
"""
将 ct_dma 目录中的 pickle 文件转换为 CSV 格式
用于更方便地查看和分析 AIS 轨迹数据
"""

import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pickle_data(file_path):
    """
    加载 pickle 文件数据
    
    参数:
        file_path: pickle 文件路径
        
    返回:
        data: 加载的数据
    """
    logger.info(f"正在加载文件: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"成功加载 {len(data)} 条记录")
    return data


def trajectory_to_dataframe(trajectories, dataset_name):
    """
    将轨迹数据转换为 DataFrame
    
    参数:
        trajectories: 轨迹数据列表
        dataset_name: 数据集名称
        
    返回:
        df: 包含所有轨迹点的 DataFrame
    """
    all_points = []
    
    for traj_idx, traj_data in enumerate(trajectories):
        if 'traj' not in traj_data:
            continue
            
        traj = traj_data['traj']
        
        # 获取 MMSI（如果存在）
        mmsi = traj_data.get('mmsi', f'unknown_{traj_idx}')
        
        # 处理每个轨迹点
        for point_idx, point in enumerate(traj):
            if len(point) >= 4:  # 确保有足够的数据列
                point_data = {
                    'dataset': dataset_name,
                    'trajectory_id': traj_idx,
                    'mmsi': mmsi,
                    'point_index': point_idx,
                    'latitude': point[0],
                    'longitude': point[1],
                    'sog': point[2],  # Speed Over Ground
                    'cog': point[3],  # Course Over Ground
                }
                
                # 如果有更多列，添加它们
                if len(point) > 4:
                    for i in range(4, len(point)):
                        point_data[f'extra_col_{i}'] = point[i]
                
                all_points.append(point_data)
    
    df = pd.DataFrame(all_points)
    logger.info(f"转换完成: {len(df)} 个轨迹点")
    return df


def coastline_to_dataframe(coastline_data):
    """
    将海岸线数据转换为 DataFrame
    
    参数:
        coastline_data: 海岸线多边形数据
        
    返回:
        df: 包含海岸线点的 DataFrame
    """
    all_points = []
    
    # 处理不同可能的数据结构
    if isinstance(coastline_data, list):
        for poly_idx, polygon in enumerate(coastline_data):
            if hasattr(polygon, 'exterior'):
                # Shapely Polygon 对象
                coords = list(polygon.exterior.coords)
                for point_idx, (lon, lat) in enumerate(coords):
                    all_points.append({
                        'polygon_id': poly_idx,
                        'point_index': point_idx,
                        'longitude': lon,
                        'latitude': lat,
                        'type': 'exterior'
                    })
            elif isinstance(polygon, (list, tuple, np.ndarray)):
                # 坐标数组
                for point_idx, point in enumerate(polygon):
                    if len(point) >= 2:
                        all_points.append({
                            'polygon_id': poly_idx,
                            'point_index': point_idx,
                            'longitude': point[0],
                            'latitude': point[1],
                            'type': 'coastline'
                        })
    
    df = pd.DataFrame(all_points)
    logger.info(f"海岸线转换完成: {len(df)} 个点")
    return df


def convert_pkl_to_csv(data_dir, output_dir):
    """
    将 ct_dma 目录中的所有 pickle 文件转换为 CSV
    
    参数:
        data_dir: 输入数据目录
        output_dir: 输出 CSV 目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义文件映射
    files_to_convert = {
        'ct_dma_train.pkl': 'train_trajectories.csv',
        'ct_dma_valid.pkl': 'valid_trajectories.csv', 
        'ct_dma_test.pkl': 'test_trajectories.csv',
        'dma_coastline_polygons.pkl': 'coastline_polygons.csv'
    }
    
    summary_info = []
    
    for pkl_file, csv_file in files_to_convert.items():
        pkl_path = os.path.join(data_dir, pkl_file)
        csv_path = os.path.join(output_dir, csv_file)
        
        if not os.path.exists(pkl_path):
            logger.warning(f"文件不存在: {pkl_path}")
            continue
        
        try:
            # 加载数据
            data = load_pickle_data(pkl_path)
            
            # 根据文件类型转换
            if 'coastline' in pkl_file:
                df = coastline_to_dataframe(data)
                data_type = 'coastline'
            else:
                dataset_name = pkl_file.replace('.pkl', '').replace('ct_dma_', '')
                df = trajectory_to_dataframe(data, dataset_name)
                data_type = 'trajectory'
            
            # 保存 CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"已保存: {csv_path}")
            
            # 记录摘要信息
            summary_info.append({
                'source_file': pkl_file,
                'output_file': csv_file,
                'data_type': data_type,
                'records_count': len(df),
                'file_size_mb': round(os.path.getsize(csv_path) / (1024*1024), 2)
            })
            
        except Exception as e:
            logger.error(f"转换 {pkl_file} 时出错: {str(e)}")
            continue
    
    # 创建转换摘要
    summary_df = pd.DataFrame(summary_info)
    summary_path = os.path.join(output_dir, 'conversion_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    
    logger.info("=" * 50)
    logger.info("转换摘要:")
    print(summary_df.to_string(index=False))
    logger.info(f"摘要已保存到: {summary_path}")


def analyze_data_structure(data_dir):
    """
    分析数据结构，输出详细信息
    
    参数:
        data_dir: 数据目录
    """
    logger.info("=" * 50)
    logger.info("数据结构分析:")
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_dir, filename)
            logger.info(f"\n--- {filename} ---")
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                logger.info(f"数据类型: {type(data)}")
                logger.info(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    logger.info(f"第一项类型: {type(first_item)}")
                    
                    if isinstance(first_item, dict):
                        logger.info(f"字典键: {list(first_item.keys())}")
                        if 'traj' in first_item:
                            traj = first_item['traj']
                            logger.info(f"轨迹形状: {traj.shape if hasattr(traj, 'shape') else len(traj)}")
                            if len(traj) > 0:
                                logger.info(f"轨迹点维度: {len(traj[0])}")
                                logger.info(f"示例点: {traj[0]}")
                
            except Exception as e:
                logger.error(f"分析 {filename} 时出错: {str(e)}")


def main():
    """主函数"""
    # 设置路径
    data_dir = r'c:\Users\zxf\Desktop\hjtra\new\transformer_time\data\ct_dma'
    output_dir = r'c:\Users\zxf\Desktop\hjtra\new\transformer_time\data\ct_dma_csv'
    
    logger.info("开始 AIS 数据 PKL 到 CSV 转换")
    logger.info(f"输入目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 分析数据结构
    analyze_data_structure(data_dir)
    
    # 执行转换
    convert_pkl_to_csv(data_dir, output_dir)
    
    logger.info("转换完成！")
    logger.info(f"CSV 文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()
