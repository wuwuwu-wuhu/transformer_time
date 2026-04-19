"""模型评估模块"""

import numpy as np
import logging
import torch
from tqdm import tqdm
import trainers
import utils

logger = logging.getLogger(__name__)


def evaluate_model(model, aisdls, config, init_seqlen):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        aisdls: 数据加载器字典
        config: 配置对象
        init_seqlen: 初始序列长度
        
    返回:
        dict: 包含评估结果的字典
    """
    # 设置评估参数
    v_ranges = torch.tensor([2, 3, 0, 0]).to(config.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(config.device)
    
    # 使用配置中的预测步数，如果没有则使用默认值
    prediction_steps = getattr(config, 'prediction_steps', 24)
    max_seqlen = init_seqlen + prediction_steps
    
    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    
    logger.info("Starting model evaluation...")
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]), desc="Evaluating")
    
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(config.device)
            masks = masks[:, :max_seqlen].to(config.device)
            batchsize = seqs.shape[0]
            
            # 初始化误差张量
            error_ens = torch.zeros((batchsize, max_seqlen - config.init_seqlen, config.n_samples)).to(config.device)
            
            # 多次采样评估
            for i_sample in range(config.n_samples):
                # 生成预测
                preds = trainers.sample(
                    model,
                    seqs_init,
                    max_seqlen - init_seqlen,
                    temperature=1.0,
                    sample=True,
                    sample_mode=config.sample_mode,
                    r_vicinity=config.r_vicinity,
                    top_k=config.top_k
                )
                
                # 计算预测误差
                inputs = seqs[:, :max_seqlen, :].to(config.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, config.init_seqlen:]
            
            # 累积批次结果
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, config.init_seqlen:])
    
    # 计算最终误差统计
    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()
    
    # 计算平均误差统计
    l_mean_vals = [x for x in l_mean_errors]
    mean_errors = torch.cat(l_mean_vals, dim=0) * m_masks
    mean_pred_errors = mean_errors.sum(dim=0) / m_masks.sum(dim=0)
    mean_pred_errors = mean_pred_errors.detach().cpu().numpy()
    
    logger.info("Model evaluation completed!")
    
    # 返回评估结果
    results = {
        'min_errors': pred_errors,
        'mean_errors': mean_pred_errors,
        'time_steps': np.arange(len(pred_errors)) / 6,  # 转换为小时
        'masks': m_masks.detach().cpu().numpy()
    }
    
    return results


def print_evaluation_summary(results):
    """
    打印评估结果摘要
    
    参数:
        results: 评估结果字典
    """
    min_errors = results['min_errors']
    mean_errors = results['mean_errors']
    time_steps = results['time_steps']
    
    logger.info("=== Evaluation Summary ===")
    logger.info(f"Total time steps evaluated: {len(min_errors)}")
    logger.info(f"Time range: 0 - {time_steps[-1]:.1f} hours")
    
    # 关键时间点的误差
    key_timesteps = [6, 12, 18]  # 对应 1, 2, 3 小时
    for ts in key_timesteps:
        if ts < len(min_errors):
            hour = ts / 6
            logger.info(f"Error at {hour:.1f}h: Min={min_errors[ts]:.4f}km, Mean={mean_errors[ts]:.4f}km")
    
    # 整体统计
    logger.info(f"Overall min error: {np.min(min_errors):.4f}km")
    logger.info(f"Overall max error: {np.max(min_errors):.4f}km")
    logger.info(f"Overall mean error: {np.mean(min_errors):.4f}km")


def save_evaluation_results(results, config):
    """
    保存评估结果到文件
    
    参数:
        results: 评估结果字典
        config: 配置对象
    """
    import pickle
    import os
    
    results_path = os.path.join(config.savedir, "evaluation_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # 同时保存为 numpy 格式便于后续分析
    np_results_path = os.path.join(config.savedir, "evaluation_results.npz")
    np.savez(
        np_results_path,
        min_errors=results['min_errors'],
        mean_errors=results['mean_errors'],
        time_steps=results['time_steps']
    )
    logger.info(f"Evaluation results (numpy) saved to {np_results_path}")


def run_evaluation_pipeline(model, aisdls, config, init_seqlen):
    """
    运行完整的评估流水线
    
    参数:
        model: 训练好的模型
        aisdls: 数据加载器字典
        config: 配置对象
        init_seqlen: 初始序列长度
        
    返回:
        dict: 评估结果
    """
    # 执行评估
    results = evaluate_model(model, aisdls, config, init_seqlen)
    
    # 打印摘要
    print_evaluation_summary(results)
    
    # 保存结果
    save_evaluation_results(results, config)
    
    return results