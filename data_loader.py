"""数据加载和预处理模块"""

import numpy as np
import os
import pickle
import logging
from torch.utils.data import DataLoader
import datasets

logger = logging.getLogger(__name__)


def load_ais_data(config):
    """
    加载 AIS 数据集并创建 PyTorch DataLoader
    
    参数:
        config: 配置对象，包含数据路径和参数
        
    返回:
        tuple: (Data, aisdatasets, aisdls) 包含原始数据、数据集对象和数据加载器
    """
    moving_threshold = 0.05
    l_pkl_filenames = [config.trainset_name, config.validset_name, config.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(config.datadir, filename)
        logger.info(f"Loading {datapath}...")
        print(f"Loading {datapath}...")
        
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
            
        # 预处理：移除静止部分
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
            
        # 过滤无效数据
        Data[phase] = [x for x in l_pred_errors 
                      if not np.isnan(x["traj"]).any() and len(x["traj"]) > config.min_seqlen]
        
        print(f"Original: {len(l_pred_errors)}, Filtered: {len(Data[phase])}")
        logger.info(f"Phase {phase}: {len(Data[phase])} samples")
        
        print("Creating pytorch dataset...")
        
        # 创建数据集
        # max_seqlen = config.max_seqlen + 1 因为后续会使用 inputs = x[:-1], targets = x[1:]
        if config.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(
                Data[phase],
                max_seqlen=config.max_seqlen + 1,
                device=config.device
            )
        else:
            aisdatasets[phase] = datasets.AISDataset(
                Data[phase],
                max_seqlen=config.max_seqlen + 1,
                device=config.device
            )
            
        # 创建数据加载器
        shuffle = phase != "test"  # 测试集不打乱
        aisdls[phase] = DataLoader(
            aisdatasets[phase],
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 0
        )
    
    # 更新配置中的 final_tokens
    config.final_tokens = 2 * len(aisdatasets["train"]) * config.max_seqlen
    
    return Data, aisdatasets, aisdls


def get_data_info(aisdatasets):
    """
    获取数据集信息
    
    参数:
        aisdatasets: 数据集字典
        
    返回:
        dict: 包含各阶段数据集大小的字典
    """
    info = {}
    for phase in aisdatasets:
        info[phase] = len(aisdatasets[phase])
    return info