"""结果可视化和画图模块"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)


def plot_prediction_errors(results, config, save_path=None):
    """
    绘制预测误差曲线
    
    参数:
        results: 评估结果字典
        config: 配置对象
        save_path: 保存路径，如果为 None 则使用默认路径
        
    返回:
        str: 保存的图片路径
    """
    pred_errors = results['min_errors']
    v_times = results['time_steps']
    
    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(v_times, pred_errors, 'b-', linewidth=2, label='Prediction Error')
    
    # 标记关键时间点
    key_timesteps = [6, 12, 18]  # 对应 1, 2, 3 小时的索引
    key_hours = [1, 2, 3]
    colors = ['red', 'green', 'orange']
    
    for i, (timestep, hour, color) in enumerate(zip(key_timesteps, key_hours, colors)):
        if timestep < len(pred_errors):
            # 绘制垂直线
            plt.plot(hour, pred_errors[timestep], "o", color=color, markersize=8)
            plt.plot([hour, hour], [0, pred_errors[timestep]], color=color, linestyle='--', alpha=0.7)
            plt.plot([0, hour], [pred_errors[timestep], pred_errors[timestep]], color=color, linestyle='--', alpha=0.7)
            
            # 添加数值标注
            plt.text(hour + 0.12, pred_errors[timestep] - 0.5, 
                    f"{pred_errors[timestep]:.4f}", 
                    fontsize=10, color=color, fontweight='bold')
    
    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Prediction errors (km)", fontsize=12)
    plt.title("AIS Trajectory Prediction Error Over Time", fontsize=14, fontweight='bold')
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图片
    if save_path is None:
        save_path = os.path.join(config.savedir, "prediction_error.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction error plot saved to {save_path}")
    return save_path


def plot_error_comparison(results, config, save_path=None):
    """
    绘制最小误差和平均误差的对比图
    
    参数:
        results: 评估结果字典
        config: 配置对象
        save_path: 保存路径
        
    返回:
        str: 保存的图片路径
    """
    min_errors = results['min_errors']
    mean_errors = results['mean_errors']
    v_times = results['time_steps']
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    plt.plot(v_times, min_errors, 'b-', linewidth=2, label='Min Error', alpha=0.8)
    plt.plot(v_times, mean_errors, 'r-', linewidth=2, label='Mean Error', alpha=0.8)
    
    plt.fill_between(v_times, min_errors, mean_errors, alpha=0.2, color='gray', label='Error Range')
    
    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Prediction errors (km)", fontsize=12)
    plt.title("Min vs Mean Prediction Errors", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path is None:
        save_path = os.path.join(config.savedir, "error_comparison.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Error comparison plot saved to {save_path}")
    return save_path


def denormalize_coordinates(coords, config):
    """
    将归一化坐标转换回真实地理坐标
    
    参数:
        coords: 归一化坐标 [lat, lon, sog, cog]
        config: 配置对象
        
    返回:
        denorm_coords: 真实坐标
    """
    denorm_coords = coords.copy()
    if len(coords.shape) == 2 and coords.shape[1] >= 2:
        # 反归一化纬度和经度
        denorm_coords[:, 0] = coords[:, 0] * (config.lat_max - config.lat_min) + config.lat_min
        denorm_coords[:, 1] = coords[:, 1] * (config.lon_max - config.lon_min) + config.lon_min
    return denorm_coords


def plot_trajectory_samples(model, aisdls, config, init_seqlen, n_plots=7, save_path=None):
    """
    为每个轨迹单独绘制预测样本图
    
    参数:
        model: 训练好的模型
        aisdls: 数据加载器字典
        config: 配置对象
        init_seqlen: 初始序列长度
        n_plots: 绘制的轨迹数量
        save_path: 保存路径（目录）
        
    返回:
        list: 保存的图片路径列表
    """
    import torch
    import trainers
    from datetime import datetime
    
    # 获取测试数据
    seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
    seqs_init = seqs[:n_plots, :init_seqlen, :].to(config.device)
    
    # 生成预测
    model.eval()
    with torch.no_grad():
        preds = trainers.sample(
            model,
            seqs_init,
            96 - init_seqlen,
            temperature=1.0,
            sample=True,
            sample_mode=config.sample_mode,
            r_vicinity=config.r_vicinity,
            top_k=config.top_k
        )
    
    preds_np = preds.detach().cpu().numpy()
    inputs_np = seqs.detach().cpu().numpy()
    
    # 创建保存目录
    if save_path is None:
        save_dir = os.path.join(config.savedir, "trajectory_plots")
    else:
        save_dir = save_path
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = []
    
    # 为每个轨迹单独绘图
    for idx in range(n_plots):
        try:
            seqlen = seqlens[idx].item()
            mmsi = mmsis[idx] if mmsis is not None else f"unknown_{idx}"
            time_start = time_starts[idx] if time_starts is not None else "unknown_time"
            
            # 创建新图
            plt.figure(figsize=(12, 8), dpi=150)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 获取轨迹数据并反归一化
            historical_traj = denormalize_coordinates(inputs_np[idx][:init_seqlen], config)  # 历史轨迹
            true_traj = denormalize_coordinates(inputs_np[idx][init_seqlen:seqlen], config)  # 真实轨迹
            pred_traj = denormalize_coordinates(preds_np[idx][init_seqlen:], config)  # 预测轨迹
            
            # 绘制历史轨迹（黄色）
            if len(historical_traj) > 0:
                plt.plot(historical_traj[:, 1], historical_traj[:, 0], 
                        'o-', color='gold', linewidth=3, markersize=6, 
                        label='历史轨迹', alpha=0.8)
            
            # 绘制真实轨迹（红色）
            if len(true_traj) > 0:
                plt.plot(true_traj[:, 1], true_traj[:, 0], 
                        'o-', color='red', linewidth=3, markersize=6, 
                        label='真实轨迹')
            
            # 绘制预测轨迹（蓝色虚线）
            if len(pred_traj) > 0:
                plt.plot(pred_traj[:, 1], pred_traj[:, 0], 
                        'o--', color='blue', linewidth=3, markersize=6, 
                        label='预测轨迹', alpha=0.8)
            
            # 设置图表样式
            plt.grid(True, alpha=0.3, linewidth=0.8)
            plt.xlabel('经度 (°)', fontsize=14, fontweight='bold')
            plt.ylabel('纬度 (°)', fontsize=14, fontweight='bold')
            
            # 设置标题
            title = f'真实轨迹与预测轨迹 (MMSI: {mmsi}_{time_start})'
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            # 设置图例
            plt.legend(fontsize=12, loc='upper right', framealpha=0.9,
                      fancybox=True, shadow=True)
            
            # 自动调整坐标轴范围
            all_lons = np.concatenate([historical_traj[:, 1], true_traj[:, 1], pred_traj[:, 1]])
            all_lats = np.concatenate([historical_traj[:, 0], true_traj[:, 0], pred_traj[:, 0]])
            
            if len(all_lons) > 0 and len(all_lats) > 0:
                lon_range = all_lons.max() - all_lons.min()
                lat_range = all_lats.max() - all_lats.min()
                
                # 设置最小范围，避免坐标轴过于紧密
                min_range = 0.01  # 最小显示范围
                lon_margin = max(lon_range * 0.1, min_range)
                lat_margin = max(lat_range * 0.1, min_range)
                
                # 如果数据范围太小，使用固定的显示范围
                if lon_range < min_range:
                    lon_center = (all_lons.max() + all_lons.min()) / 2
                    plt.xlim(lon_center - min_range, lon_center + min_range)
                else:
                    plt.xlim(all_lons.min() - lon_margin, all_lons.max() + lon_margin)
                
                if lat_range < min_range:
                    lat_center = (all_lats.max() + all_lats.min()) / 2
                    plt.ylim(lat_center - min_range, lat_center + min_range)
                else:
                    plt.ylim(all_lats.min() - lat_margin, all_lats.max() + lat_margin)
                
                # 添加坐标信息到图上
                coord_text = f'经度: {all_lons.min():.4f}° - {all_lons.max():.4f}°\n纬度: {all_lats.min():.4f}° - {all_lats.max():.4f}°'
                plt.text(0.02, 0.02, coord_text, transform=plt.gca().transAxes,
                        fontsize=9, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 保存图片
            filename = f"trajectory_{idx+1}_{mmsi}_{time_start}.png"
            file_path = os.path.join(save_dir, filename)
            plt.savefig(file_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            saved_files.append(file_path)
            logger.info(f"Trajectory plot {idx+1} saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error plotting trajectory {idx+1}: {str(e)}")
            plt.close()
            continue
    
    logger.info(f"Generated {len(saved_files)} individual trajectory plots")
    return saved_files


def create_summary_report(results, config, save_path=None):
    """
    创建评估结果的文本摘要报告
    
    参数:
        results: 评估结果字典
        config: 配置对象
        save_path: 保存路径
        
    返回:
        str: 保存的报告路径
    """
    min_errors = results['min_errors']
    mean_errors = results['mean_errors']
    time_steps = results['time_steps']
    
    if save_path is None:
        save_path = os.path.join(config.savedir, "evaluation_report.txt")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== TrAISformer 评估报告 ===\n\n")
        f.write(f"模型配置:\n")
        f.write(f"  - 数据集: {config.dataset_name}\n")
        f.write(f"  - 模式: {config.mode}\n")
        f.write(f"  - 采样模式: {config.sample_mode}\n")
        f.write(f"  - 批次大小: {config.batch_size}\n")
        f.write(f"  - 学习率: {config.learning_rate}\n")
        f.write(f"  - 序列长度: {config.init_seqlen}-{config.max_seqlen}\n\n")
        
        f.write(f"评估结果:\n")
        f.write(f"  - 总时间步数: {len(min_errors)}\n")
        f.write(f"  - 时间范围: 0 - {time_steps[-1]:.1f} 小时\n\n")
        
        f.write("关键时间点误差:\n")
        key_timesteps = [6, 12, 18]
        for ts in key_timesteps:
            if ts < len(min_errors):
                hour = ts / 6
                f.write(f"  - {hour:.1f}小时: 最小误差={min_errors[ts]:.4f}km, 平均误差={mean_errors[ts]:.4f}km\n")
        
        f.write(f"\n整体统计:\n")
        f.write(f"  - 最小误差: {np.min(min_errors):.4f}km\n")
        f.write(f"  - 最大误差: {np.max(min_errors):.4f}km\n")
        f.write(f"  - 平均误差: {np.mean(min_errors):.4f}km\n")
        f.write(f"  - 标准差: {np.std(min_errors):.4f}km\n")
    
    logger.info(f"Evaluation report saved to {save_path}")
    return save_path


def run_visualization_pipeline(results, model, aisdls, config, init_seqlen):
    """
    运行完整的可视化流水线
    
    参数:
        results: 评估结果字典
        model: 训练好的模型
        aisdls: 数据加载器字典
        config: 配置对象
        init_seqlen: 初始序列长度
        
    返回:
        dict: 包含所有生成文件路径的字典
    """
    logger.info("Starting visualization pipeline...")
    
    generated_files = {}
    
    # 绘制预测误差曲线
    generated_files['prediction_error'] = plot_prediction_errors(results, config)
    
    # 绘制误差对比图
    generated_files['error_comparison'] = plot_error_comparison(results, config)
    
    # 绘制轨迹样本
    generated_files['trajectory_samples'] = plot_trajectory_samples(
        model, aisdls, config, init_seqlen
    )
    
    # 创建摘要报告
    generated_files['summary_report'] = create_summary_report(results, config)
    
    logger.info("Visualization pipeline completed!")
    logger.info("Generated files:")
    for name, path in generated_files.items():
        logger.info(f"  - {name}: {path}")
    
    return generated_files