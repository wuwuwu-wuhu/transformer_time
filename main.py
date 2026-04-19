"""
TrAISformer 主入口文件
整合数据加载、训练、评估和可视化模块

使用方法:
    python main.py                    # 运行完整流水线
    python main.py --train-only       # 仅训练
    python main.py --eval-only        # 仅评估
    python main.py --viz-only         # 仅可视化
"""

import argparse
import logging
import torch
import utils
from config_transformer import Config

# 导入各个模块
import data_loader
import train
import evaluate
import visualize

logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TrAISformer - AIS Trajectory Prediction')
    
    parser.add_argument('--train-only', action='store_true', 
                       help='Only run training phase')
    parser.add_argument('--eval-only', action='store_true', 
                       help='Only run evaluation phase')
    parser.add_argument('--viz-only', action='store_true', 
                       help='Only run visualization phase')
    parser.add_argument('--no-train', action='store_true', 
                       help='Skip training phase')
    parser.add_argument('--no-eval', action='store_true', 
                       help='Skip evaluation phase')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization phase')
    
    return parser.parse_args()


def setup_environment():
    """设置运行环境"""
    # 设置随机种子
    utils.set_seed(42)
    
    # 设置 torch.pi
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    
    # 基础日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置环境
    setup_environment()
    
    # 加载配置
    config = Config()
    init_seqlen = config.init_seqlen
    
    logger.info("=== TrAISformer Pipeline Started ===")
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Save directory: {config.savedir}")
    
    # 确定运行阶段
    run_data_loading = True
    run_training = not args.no_train and not args.eval_only and not args.viz_only
    run_evaluation = not args.no_eval and not args.train_only
    run_visualization = not args.no_viz and not args.train_only
    
    if args.train_only:
        run_evaluation = False
        run_visualization = False
    elif args.eval_only:
        run_training = False
        run_visualization = False
    elif args.viz_only:
        run_training = False
        run_evaluation = False
        # 可视化需要数据加载器，所以保持 run_data_loading = True
    
    # 阶段1: 数据加载
    if run_data_loading:
        logger.info("=== Phase 1: Data Loading ===")
        Data, aisdatasets, aisdls = data_loader.load_ais_data(config)
        data_info = data_loader.get_data_info(aisdatasets)
        logger.info(f"Data loaded: {data_info}")
    
    # 阶段2: 模型训练
    model, trainer = None, None
    if run_training:
        logger.info("=== Phase 2: Model Training ===")
        model, trainer = train.run_training_pipeline(config, aisdatasets, aisdls, init_seqlen)
    
    # 阶段3: 模型评估
    results = None
    if run_evaluation:
        logger.info("=== Phase 3: Model Evaluation ===")
        
        # 如果没有训练，需要加载模型
        if model is None:
            model = train.create_model(config)
            model = train.load_trained_model(model, config)
            model = model.to(config.device)  # 确保模型在正确的设备上
        
        results = evaluate.run_evaluation_pipeline(model, aisdls, config, init_seqlen)
    
    # 阶段4: 结果可视化
    if run_visualization:
        logger.info("=== Phase 4: Visualization ===")
        
        # 如果没有评估，尝试加载评估结果
        if results is None:
            try:
                import pickle
                import os
                results_path = os.path.join(config.savedir, "evaluation_results.pkl")
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                logger.info(f"Loaded evaluation results from {results_path}")
            except FileNotFoundError:
                logger.error("No evaluation results found. Please run evaluation first.")
                return
        
        # 如果没有模型，需要加载模型（用于轨迹可视化）
        if model is None:
            model = train.create_model(config)
            model = train.load_trained_model(model, config)
            model = model.to(config.device)  # 确保模型在正确的设备上
        
        generated_files = visualize.run_visualization_pipeline(
            results, model, aisdls, config, init_seqlen
        )
    
    logger.info("=== TrAISformer Pipeline Completed ===")
    
    # 打印摘要
    if run_evaluation and results is not None:
        logger.info("=== Final Results Summary ===")
        min_errors = results['min_errors']
        key_timesteps = [6, 12, 18]
        for ts in key_timesteps:
            if ts < len(min_errors):
                hour = ts / 6
                logger.info(f"Error at {hour:.1f}h: {min_errors[ts]:.4f}km")


if __name__ == "__main__":
    main()