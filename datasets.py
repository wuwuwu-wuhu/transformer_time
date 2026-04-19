"""自定义的 PyTorch 数据集.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class AISDataset(Dataset):
    """自定义 AIS 数据集（位置-速度-航向序列）。
    """
    
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        参数：
            l_data: 由字典组成的列表，每个元素是一条 AIS 轨迹。
                - l_data[idx]["mmsi"]: 船舶的 MMSI 标识。
                - l_data[idx]["traj"]: (N, 5) 的矩阵，列为 [LAT, LON, SOG, COG, TIMESTAMP]
                  其中 lat/lon/sog/cog 已标准化到 [0,1) 区间。
            max_seqlen: 可选，序列最大长度（超过将被截断，不足会在尾部补零）。
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """按索引获取一个样本。
        
        返回：
            seq: 张量，形状 (max_seqlen, 4)，列为 [lat, lon, sog, cog]（已归一化）。
            mask: 张量，形状 (max_seqlen, )，若为 padding 位置则为 0，否则为 1。
            seqlen: 实际有效长度（int）。
            mmsi: 船舶 MMSI（int）。
            time_start: 该轨迹起始的 Unix 时间戳（int）。
        """
        
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        seq[:seqlen,:] = m_v[:seqlen,:]
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        
        return seq , mask, seqlen, mmsi, time_start
    
class AISDataset_grad(Dataset):
    """自定义 AIS 数据集。
    返回位置以及位置的梯度（dlat/dlon）表示。
    """
    
    def __init__(self, 
                 l_data, 
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        参数：
            l_data: 由字典组成的列表，每个元素是一条 AIS 轨迹。
            dlat_max, dlon_max: 位置梯度（相邻两时刻差分）的最大值，用于归一化梯度到 [0,1)。
                例如 dlat_max = max(lat[i+1] - lat[i])。
            max_seqlen: 可选，序列最大长度。
        """    
            
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        # lat and lon
        seq[:seqlen,:2] = m_v[:seqlen,:2] 
        # dlat and dlon
        dpos = (m_v[1:,:2]-m_v[:-1,:2]+self.dpos_max )/(2*self.dpos_max )
        dpos = np.concatenate((dpos[:1,:],dpos),axis=0)
        dpos[dpos>=1] = 0.9999
        dpos[dpos<=0] = 0.0
        seq[:seqlen,2:] = dpos[:seqlen,:2] 
        
        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        
        return seq , mask, seqlen, mmsi, time_start