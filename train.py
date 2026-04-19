"""模型训练模块"""

import os
import logging
import torch
import models
import trainers
import utils

logger = logging.getLogger(__name__)


def setup_logging(config):
    """
    设置日志系统
    
    参数:
        config: 配置对象
    """
    if not os.path.isdir(config.savedir):
        os.makedirs(config.savedir)
        print('======= Create directory to store trained models: ' + config.savedir)
    else:
        print('======= Directory to store trained models: ' + config.savedir)
    
    utils.new_log(config.savedir, "log")


def create_model(config):
    """
    创建模型实例
    
    参数:
        config: 配置对象
        
    返回:
        model: TrAISformer 模型实例
    """
    model = models.TrAISformer(config, partition_model=None)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):.2e} parameters")
    return model


def train_model(model, aisdatasets, aisdls, config, init_seqlen):
    """
    训练模型
    
    参数:
        model: 模型实例
        aisdatasets: 数据集字典
        aisdls: 数据加载器字典
        config: 配置对象
        init_seqlen: 初始序列长度
        
    返回:
        trainer: 训练器实例
    """
    # 创建训练器
    trainer = trainers.Trainer(
        model, 
        aisdatasets["train"], 
        aisdatasets["valid"], 
        config, 
        savedir=config.savedir, 
        device=config.device, 
        aisdls=aisdls, 
        INIT_SEQLEN=init_seqlen
    )
    
    # 开始训练
    if config.retrain:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
    else:
        logger.info("Skipping training (retrain=False)")
    
    return trainer


def load_trained_model(model, config):
    """
    加载已训练的模型权重
    
    参数:
        model: 模型实例
        config: 配置对象
        
    返回:
        model: 加载权重后的模型
    """
    if os.path.exists(config.ckpt_path):
        logger.info(f"Loading trained model from {config.ckpt_path}")
        model.load_state_dict(torch.load(config.ckpt_path, map_location=config.device))
        logger.info("Model loaded successfully!")
    else:
        logger.warning(f"Model checkpoint not found at {config.ckpt_path}")
        logger.warning("Using randomly initialized model")
    
    return model


def setup_tensorboard(config):
    """
    设置 TensorBoard 日志（如果启用）
    
    参数:
        config: 配置对象
        
    返回:
        tb: TensorBoard SummaryWriter 实例或 None
    """
    if config.tb_log:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(config.savedir, 'tensorboard'))
            logger.info("TensorBoard logging enabled")
            return tb
        except ImportError:
            logger.warning("TensorBoard not available, skipping TB logging")
            return None
    return None


def run_training_pipeline(config, aisdatasets, aisdls, init_seqlen):
    """
    运行完整的训练流水线
    
    参数:
        config: 配置对象
        aisdatasets: 数据集字典
        aisdls: 数据加载器字典
        init_seqlen: 初始序列长度
        
    返回:
        tuple: (model, trainer) 训练后的模型和训练器
    """
    # 设置日志
    setup_logging(config)
    
    # 设置 TensorBoard
    tb = setup_tensorboard(config)
    
    # 创建模型
    model = create_model(config)
    
    # 训练模型
    trainer = train_model(model, aisdatasets, aisdls, config, init_seqlen)
    
    # 加载最佳模型权重
    model = load_trained_model(model, config)
    
    return model, trainer