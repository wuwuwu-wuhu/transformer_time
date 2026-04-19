# coding=utf-8
"""
丹麦海域海岸线可视化工具
绘制 DMA (Danish Maritime Authority) 海岸线多边形数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_coastline_data(file_path):
    """
    加载海岸线数据
    
    参数:
        file_path: 海岸线数据文件路径
        
    返回:
        coastline_data: 海岸线多边形数据列表
    """
    logger.info(f"加载海岸线数据: {file_path}")
    
    with open(file_path, 'rb') as f:
        coastline_data = pickle.load(f)
    
    logger.info(f"加载完成，包含 {len(coastline_data)} 个海岸线多边形")
    return coastline_data


def analyze_coastline_data(coastline_data):
    """
    分析海岸线数据的基本信息
    
    参数:
        coastline_data: 海岸线数据
        
    返回:
        dict: 分析结果
    """
    total_polygons = len(coastline_data)
    total_points = sum(len(polygon) for polygon in coastline_data)
    
    # 计算整体边界
    all_lons = []
    all_lats = []
    
    for polygon in coastline_data:
        if len(polygon) > 0:
            all_lons.extend(polygon[:, 0])
            all_lats.extend(polygon[:, 1])
    
    analysis = {
        'total_polygons': total_polygons,
        'total_points': total_points,
        'lon_range': (min(all_lons), max(all_lons)),
        'lat_range': (min(all_lats), max(all_lats)),
        'polygon_sizes': [len(polygon) for polygon in coastline_data]
    }
    
    logger.info(f"海岸线分析结果:")
    logger.info(f"  多边形数量: {analysis['total_polygons']}")
    logger.info(f"  总点数: {analysis['total_points']}")
    logger.info(f"  经度范围: {analysis['lon_range'][0]:.4f} - {analysis['lon_range'][1]:.4f}")
    logger.info(f"  纬度范围: {analysis['lat_range'][0]:.4f} - {analysis['lat_range'][1]:.4f}")
    
    return analysis


def plot_coastline_overview(coastline_data, save_path=None):
    """
    绘制海岸线总览图
    
    参数:
        coastline_data: 海岸线数据
        save_path: 保存路径
        
    返回:
        str: 保存的图片路径
    """
    logger.info("绘制海岸线总览图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 12), dpi=150)
    
    # 绘制每个海岸线多边形
    colors = plt.cm.Set3(np.linspace(0, 1, len(coastline_data)))
    
    for i, polygon in enumerate(coastline_data):
        if len(polygon) > 0:
            # 绘制多边形边界
            ax.plot(polygon[:, 0], polygon[:, 1], 
                   color=colors[i], linewidth=1.5, alpha=0.8,
                   label=f'海岸线 {i+1}' if i < 5 else "")
            
            # 填充多边形（可选）
            ax.fill(polygon[:, 0], polygon[:, 1], 
                   color=colors[i], alpha=0.3)
    
    # 设置图表样式
    ax.set_xlabel('经度 (°)', fontsize=14, fontweight='bold')
    ax.set_ylabel('纬度 (°)', fontsize=14, fontweight='bold')
    ax.set_title('丹麦海域海岸线分布图 (DMA Coastline)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # 设置坐标轴比例
    ax.set_aspect('equal', adjustable='box')
    
    # 添加图例（只显示前5个）
    if len(coastline_data) <= 5:
        ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    total_polygons = len(coastline_data)
    total_points = sum(len(polygon) for polygon in coastline_data)
    
    info_text = f'多边形数量: {total_polygons}\n总坐标点数: {total_points}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    if save_path is None:
        save_path = "coastline_overview.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"海岸线总览图已保存: {save_path}")
    return save_path


def plot_coastline_detailed(coastline_data, save_dir=None):
    """
    绘制每个海岸线多边形的详细图
    
    参数:
        coastline_data: 海岸线数据
        save_dir: 保存目录
        
    返回:
        list: 保存的图片路径列表
    """
    logger.info("绘制详细海岸线图...")
    
    if save_dir is None:
        save_dir = "coastline_details"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    saved_files = []
    
    for i, polygon in enumerate(coastline_data):
        if len(polygon) == 0:
            continue
            
        # 创建新图
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        
        # 绘制多边形
        ax.plot(polygon[:, 0], polygon[:, 1], 
               'b-', linewidth=2, alpha=0.8, label='海岸线边界')
        ax.fill(polygon[:, 0], polygon[:, 1], 
               color='lightblue', alpha=0.5, label='陆地区域')
        
        # 标记起点和终点
        ax.plot(polygon[0, 0], polygon[0, 1], 
               'go', markersize=8, label='起点')
        ax.plot(polygon[-1, 0], polygon[-1, 1], 
               'ro', markersize=8, label='终点')
        
        # 设置图表样式
        ax.set_xlabel('经度 (°)', fontsize=12, fontweight='bold')
        ax.set_ylabel('纬度 (°)', fontsize=12, fontweight='bold')
        ax.set_title(f'海岸线多边形 {i+1} (共 {len(polygon)} 个点)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # 设置坐标轴比例
        ax.set_aspect('equal', adjustable='box')
        
        # 添加统计信息
        lon_range = polygon[:, 0].max() - polygon[:, 0].min()
        lat_range = polygon[:, 1].max() - polygon[:, 1].min()
        
        info_text = f'坐标点数: {len(polygon)}\n经度跨度: {lon_range:.4f}°\n纬度跨度: {lat_range:.4f}°'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 保存图片
        filename = f"coastline_polygon_{i+1:02d}.png"
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        saved_files.append(file_path)
        logger.info(f"海岸线多边形 {i+1} 已保存: {file_path}")
    
    logger.info(f"生成了 {len(saved_files)} 个详细海岸线图")
    return saved_files


def plot_coastline_interactive(coastline_data, save_path=None):
    """
    绘制交互式海岸线图（带缩放区域）
    
    参数:
        coastline_data: 海岸线数据
        save_path: 保存路径
        
    返回:
        str: 保存的图片路径
    """
    logger.info("绘制交互式海岸线图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig = plt.figure(figsize=(18, 12))
    
    # 主图 - 完整海岸线
    ax1 = plt.subplot(2, 2, (1, 3))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(coastline_data)))
    
    for i, polygon in enumerate(coastline_data):
        if len(polygon) > 0:
            ax1.plot(polygon[:, 0], polygon[:, 1], 
                    color=colors[i], linewidth=1.5, alpha=0.8)
            ax1.fill(polygon[:, 0], polygon[:, 1], 
                    color=colors[i], alpha=0.3)
    
    ax1.set_xlabel('经度 (°)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('纬度 (°)', fontsize=12, fontweight='bold')
    ax1.set_title('丹麦海域完整海岸线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 计算整体范围
    all_lons = []
    all_lats = []
    for polygon in coastline_data:
        if len(polygon) > 0:
            all_lons.extend(polygon[:, 0])
            all_lats.extend(polygon[:, 1])
    
    # 子图1 - 北部区域
    ax2 = plt.subplot(2, 2, 2)
    north_lat_min = np.percentile(all_lats, 70)
    
    for i, polygon in enumerate(coastline_data):
        if len(polygon) > 0:
            # 只绘制北部区域
            mask = polygon[:, 1] >= north_lat_min
            if np.any(mask):
                north_polygon = polygon[mask]
                ax2.plot(north_polygon[:, 0], north_polygon[:, 1], 
                        color=colors[i], linewidth=2, alpha=0.8)
                ax2.fill(north_polygon[:, 0], north_polygon[:, 1], 
                        color=colors[i], alpha=0.4)
    
    ax2.set_title('北部海岸线详图', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 子图2 - 南部区域
    ax3 = plt.subplot(2, 2, 4)
    south_lat_max = np.percentile(all_lats, 30)
    
    for i, polygon in enumerate(coastline_data):
        if len(polygon) > 0:
            # 只绘制南部区域
            mask = polygon[:, 1] <= south_lat_max
            if np.any(mask):
                south_polygon = polygon[mask]
                ax3.plot(south_polygon[:, 0], south_polygon[:, 1], 
                        color=colors[i], linewidth=2, alpha=0.8)
                ax3.fill(south_polygon[:, 0], south_polygon[:, 1], 
                        color=colors[i], alpha=0.4)
    
    ax3.set_title('南部海岸线详图', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path is None:
        save_path = "coastline_interactive.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"交互式海岸线图已保存: {save_path}")
    return save_path


def create_coastline_report(coastline_data, analysis, save_path=None):
    """
    创建海岸线数据分析报告
    
    参数:
        coastline_data: 海岸线数据
        analysis: 分析结果
        save_path: 保存路径
        
    返回:
        str: 报告文件路径
    """
    if save_path is None:
        save_path = "coastline_analysis_report.txt"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("丹麦海域海岸线数据分析报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("数据概览:\n")
        f.write("-" * 40 + "\n")
        f.write(f"海岸线多边形数量: {analysis['total_polygons']}\n")
        f.write(f"总坐标点数: {analysis['total_points']}\n")
        f.write(f"平均每个多边形点数: {analysis['total_points'] / analysis['total_polygons']:.1f}\n\n")
        
        f.write("地理范围:\n")
        f.write("-" * 40 + "\n")
        f.write(f"经度范围: {analysis['lon_range'][0]:.4f}° - {analysis['lon_range'][1]:.4f}°\n")
        f.write(f"纬度范围: {analysis['lat_range'][0]:.4f}° - {analysis['lat_range'][1]:.4f}°\n")
        f.write(f"经度跨度: {analysis['lon_range'][1] - analysis['lon_range'][0]:.4f}°\n")
        f.write(f"纬度跨度: {analysis['lat_range'][1] - analysis['lat_range'][0]:.4f}°\n\n")
        
        f.write("多边形详情:\n")
        f.write("-" * 40 + "\n")
        for i, size in enumerate(analysis['polygon_sizes']):
            f.write(f"多边形 {i+1}: {size} 个坐标点\n")
        
        f.write(f"\n最大多边形: {max(analysis['polygon_sizes'])} 个点\n")
        f.write(f"最小多边形: {min(analysis['polygon_sizes'])} 个点\n")
    
    logger.info(f"海岸线分析报告已保存: {save_path}")
    return save_path


def main():
    """主函数"""
    logger.info("开始海岸线数据可视化...")
    
    # 数据文件路径
    coastline_file = r'c:\Users\zxf\Desktop\hjtra\new\transformer_time\data\ct_dma\dma_coastline_polygons.pkl'
    
    # 创建输出目录
    output_dir = "coastline_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        coastline_data = load_coastline_data(coastline_file)
        
        # 分析数据
        analysis = analyze_coastline_data(coastline_data)
        
        # 生成可视化
        logger.info("生成可视化图表...")
        
        # 1. 总览图
        overview_path = os.path.join(output_dir, "coastline_overview.png")
        plot_coastline_overview(coastline_data, overview_path)
        
        # 2. 交互式图
        interactive_path = os.path.join(output_dir, "coastline_interactive.png")
        plot_coastline_interactive(coastline_data, interactive_path)
        
        # 3. 详细图（每个多边形）
        details_dir = os.path.join(output_dir, "polygon_details")
        detail_files = plot_coastline_detailed(coastline_data, details_dir)
        
        # 4. 分析报告
        report_path = os.path.join(output_dir, "coastline_analysis_report.txt")
        create_coastline_report(coastline_data, analysis, report_path)
        
        logger.info("=" * 50)
        logger.info("海岸线可视化完成！")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"生成文件:")
        logger.info(f"  - 总览图: {overview_path}")
        logger.info(f"  - 交互式图: {interactive_path}")
        logger.info(f"  - 详细图: {len(detail_files)} 个文件")
        logger.info(f"  - 分析报告: {report_path}")
        
    except Exception as e:
        logger.error(f"处理海岸线数据时出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
