
"""
测试不同预测长度的脚本
自动测试多个预测时长并生成对比报告
"""

import os
import sys
import logging
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 导入项目模块
import config_transformer
import data_loader
import train
import evaluate
import visualize

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_prediction_length(prediction_steps, prediction_hours, config, model, aisdls, init_seqlen):
    """
    测试特定预测长度的性能
    
    参数:
        prediction_steps: 预测步数
        prediction_hours: 预测小时数
        config: 配置对象
        model: 训练好的模型
        aisdls: 数据加载器
        init_seqlen: 初始序列长度
        
    返回:
        dict: 包含评估结果的字典
    """
    logger.info(f"测试预测长度: {prediction_steps} 步 ({prediction_hours:.1f} 小时)")
    
    # 临时修改配置
    original_steps = config.prediction_steps
    original_hours = config.prediction_hours
    
    config.prediction_steps = prediction_steps
    config.prediction_hours = prediction_hours
    
    try:
        # 运行评估
        results = evaluate.evaluate_model(model, aisdls, config, init_seqlen)
        
        # 提取关键指标
        min_errors = results['min_errors']
        mean_errors = results['mean_errors']
        
        # 计算不同时间点的误差
        key_timesteps = [6, 12, 18, 24, 30, 36] if prediction_steps >= 36 else [6, 12, 18, 24]
        timestep_errors = {}
        
        for ts in key_timesteps:
            if ts < len(min_errors) and ts < prediction_steps:
                hour = ts / 6
                timestep_errors[f"{hour:.1f}h"] = min_errors[ts]
        
        result_summary = {
            'prediction_steps': prediction_steps,
            'prediction_hours': prediction_hours,
            'final_error': min_errors[min(prediction_steps-1, len(min_errors)-1)],
            'mean_final_error': mean_errors[min(prediction_steps-1, len(mean_errors)-1)],
            'timestep_errors': timestep_errors,
            'full_results': results
        }
        
        logger.info(f"完成测试 {prediction_steps} 步，最终误差: {result_summary['final_error']:.4f}km")
        return result_summary
        
    except Exception as e:
        logger.error(f"测试 {prediction_steps} 步时出错: {str(e)}")
        return None
        
    finally:
        # 恢复原始配置
        config.prediction_steps = original_steps
        config.prediction_hours = original_hours


def create_comparison_report(test_results, save_dir):
    """
    创建预测长度对比报告
    
    参数:
        test_results: 测试结果列表
        save_dir: 保存目录
    """
    # 创建DataFrame用于分析
    summary_data = []
    for result in test_results:
        if result is not None:
            row = {
                '预测步数': result['prediction_steps'],
                '预测时长(小时)': result['prediction_hours'],
                '最终误差(km)': result['final_error'],
                '平均最终误差(km)': result['mean_final_error']
            }
            
            # 添加不同时间点的误差
            for time_point, error in result['timestep_errors'].items():
                row[f'误差@{time_point}'] = error
                
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # 保存CSV报告
    csv_path = os.path.join(save_dir, "prediction_length_comparison.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"对比报告已保存: {csv_path}")
    
    # 创建可视化图表
    create_comparison_plots(test_results, save_dir)
    
    # 创建文本报告
    create_text_report(df, save_dir)
    
    return csv_path


def create_comparison_plots(test_results, save_dir):
    """
    创建预测长度对比图表
    
    参数:
        test_results: 测试结果列表
        save_dir: 保存目录
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 最终误差对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    hours = [r['prediction_hours'] for r in test_results if r is not None]
    final_errors = [r['final_error'] for r in test_results if r is not None]
    mean_errors = [r['mean_final_error'] for r in test_results if r is not None]
    
    ax1.plot(hours, final_errors, 'o-', linewidth=2, markersize=8, label='最小误差')
    ax1.plot(hours, mean_errors, 's--', linewidth=2, markersize=8, label='平均误差')
    ax1.set_xlabel('预测时长 (小时)', fontsize=12)
    ax1.set_ylabel('预测误差 (km)', fontsize=12)
    ax1.set_title('不同预测时长的最终误差对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 误差随时间变化图
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_results)))
    
    for i, result in enumerate(test_results):
        if result is not None:
            min_errors = result['full_results']['min_errors']
            time_points = np.arange(len(min_errors)) / 6  # 转换为小时
            
            ax2.plot(time_points, min_errors, 'o-', 
                    color=colors[i], linewidth=2, markersize=4,
                    label=f'{result["prediction_hours"]:.1f}h预测')
    
    ax2.set_xlabel('预测时间 (小时)', fontsize=12)
    ax2.set_ylabel('预测误差 (km)', fontsize=12)
    ax2.set_title('误差随预测时间的变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "prediction_length_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"对比图表已保存: {plot_path}")


def create_text_report(df, save_dir):
    """
    创建文本格式的详细报告
    
    参数:
        df: 结果DataFrame
        save_dir: 保存目录
    """
    report_path = os.path.join(save_dir, "prediction_length_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TrAISformer 预测长度测试报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("测试结果摘要:\n")
        f.write("-" * 40 + "\n")
        for _, row in df.iterrows():
            f.write(f"预测时长: {row['预测时长(小时)']:.1f}小时 ({row['预测步数']}步)\n")
            f.write(f"  最终误差: {row['最终误差(km)']:.4f} km\n")
            f.write(f"  平均误差: {row['平均最终误差(km)']:.4f} km\n")
            
            # 添加时间点误差
            for col in df.columns:
                if col.startswith('误差@'):
                    f.write(f"  {col}: {row[col]:.4f} km\n")
            f.write("\n")
        
        f.write("\n推荐使用:\n")
        f.write("-" * 40 + "\n")
        
        # 找到最佳预测长度
        best_2h_idx = df['预测时长(小时)'].sub(2).abs().idxmin()
        best_4h_idx = df['预测时长(小时)'].sub(4).abs().idxmin()
        
        f.write(f"短期预测 (2小时): {df.loc[best_2h_idx, '最终误差(km)']:.4f} km\n")
        f.write(f"中期预测 (4小时): {df.loc[best_4h_idx, '最终误差(km)']:.4f} km\n")
        
        if len(df) > 2:
            best_long_idx = df['最终误差(km)'].idxmin()
            f.write(f"最佳性能: {df.loc[best_long_idx, '预测时长(小时)']:.1f}小时 - ")
            f.write(f"{df.loc[best_long_idx, '最终误差(km)']:.4f} km\n")
    
    logger.info(f"详细报告已保存: {report_path}")


def main():
    """主函数"""
    logger.info("开始预测长度测试...")
    
    # 加载配置
    config = config_transformer.Config()
    
    # 创建结果目录
    test_dir = os.path.join(config.savedir, "prediction_length_tests")
    os.makedirs(test_dir, exist_ok=True)
    
    # 定义测试的预测长度
    test_configurations = [
        (12, 2.0),   # 2小时
        (18, 3.0),   # 3小时
        (24, 4.0),   # 4小时
        (30, 5.0),   # 5小时
        (36, 6.0),   # 6小时
        (48, 8.0),   # 8小时
        (60, 10.0),  # 10小时
        (72, 12.0),  # 12小时
    ]
    
    logger.info(f"将测试 {len(test_configurations)} 种预测长度配置")
    
    # 加载数据
    logger.info("加载数据...")
    Data, aisdatasets, aisdls = data_loader.load_ais_data(config)
    
    # 加载模型
    logger.info("加载模型...")
    model = train.create_model(config)
    model = train.load_trained_model(model, config)
    model = model.to(config.device)
    
    # 计算初始序列长度
    init_seqlen = config.init_seqlen
    
    # 运行测试
    test_results = []
    for i, (steps, hours) in enumerate(test_configurations):
        logger.info(f"进度: {i+1}/{len(test_configurations)}")
        result = test_prediction_length(steps, hours, config, model, aisdls, init_seqlen)
        if result is not None:
            test_results.append(result)
    
    # 生成对比报告
    if test_results:
        logger.info("生成对比报告...")
        create_comparison_report(test_results, test_dir)
        
        logger.info("=" * 50)
        logger.info("测试完成！结果摘要:")
        for result in test_results:
            logger.info(f"{result['prediction_hours']:.1f}h: {result['final_error']:.4f}km")
        
        logger.info(f"详细结果保存在: {test_dir}")
    else:
        logger.error("没有成功的测试结果")


if __name__ == "__main__":
    main()
