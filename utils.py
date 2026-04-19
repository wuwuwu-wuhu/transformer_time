import numpy as np
import os
import math
import logging
import random
import datetime
import socket


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.pi = torch.acos(torch.zeros(1)).item()*2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def new_log(logdir,filename):
    """定义日志输出格式并初始化日志文件."""
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S_"+socket.gethostname()+"_"+filename+".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)   
    
def haversine(input_coords, 
               pred_coords):
    """计算 input_coords 与 pred_coords 的哈弗辛（球面）距离。
    
    参数：
        input_coords, pred_coords: 形状为 (..., N) 的张量，其中 (...,0) 与 (...,1) 分别是弧度制的纬度与经度。
    
    返回：
        球面距离（单位：公里）。
    """
    R = 6371
    lat_errors = pred_coords[...,0] - input_coords[...,0]
    lon_errors = pred_coords[...,1] - input_coords[...,1]
    a = torch.sin(lat_errors/2)**2\
        +torch.cos(input_coords[:,:,0])*torch.cos(pred_coords[:,:,0])*torch.sin(lon_errors/2)**2
    c = 2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    d = R*c
    return d

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """仅保留距离当前索引最近的一段（邻域）候选。
    
    参数：
        att_logits: 形状为 (batch_size, data_size) 的张量。
        att_idxs: 形状为 (batch_size, 1) 的张量，表示当前位置的索引。
        r_vicinity: 邻域大小（保留的候选数量）。
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0],1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity/2] = -float('Inf')
    return out