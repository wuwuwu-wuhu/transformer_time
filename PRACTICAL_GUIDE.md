# TrAISformer 实践教学指南

本指南将通过实际操作为你展示如何使用 TrAISformer 进行 AIS 轨迹预测。从环境配置到模型训练，从结果分析到进阶实验，每个环节都配有具体的命令和预期输出。

---

## 1. 环境配置与项目准备

### 1.1 环境要求

在开始之前，请确保你的系统满足以下要求：

- **操作系统**：Windows / Linux / macOS
- **Python**：3.7 或更高版本
- **CUDA**：支持 CUDA 的 GPU（推荐）
- **硬盘空间**：至少 5GB（用于数据和模型）

### 1.2 创建 conda 环境

```bash
# 克隆项目（如果还没有）
git clone <repository-url>
cd TrAISformer

# 创建 conda 环境
conda create -n traisformer python=3.9

# 激活环境
conda activate traisformer

# 安装 PyTorch（GPU 版本）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy matplotlib tqdm pickle5
```

**预期输出**：
```
Collecting package metadata (current_repodata.json) ... done
Solving environment: done
...
# To activate this environment, use
#
#     $ conda activate traisformer
#
# To deactivate an active environment, use
#
#     $ conda deactivate traisformer
```

### 1.3 验证安装

```bash
# 验证 Python 版本
python --version

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**预期输出**：
```
Python 3.9.18
PyTorch: 2.0.1
CUDA: True
```

### 1.4 项目结构概览

```
TrAISformer/
├── data/                     # 数据目录
│   └── ct_dma/              # AIS 数据集
│       ├── ct_dma_train.pkl
│       ├── ct_dma_valid.pkl
│       └── ct_dma_test.pkl
├── results/                  # 训练结果输出目录
├── config_transformer.py      # 配置文件
├── main.py                   # 主入口
├── models.py                 # 模型定义
├── data_loader.py            # 数据加载
├── datasets.py               # 数据集类
├── train.py                  # 训练模块
├── trainers.py               # 训练器
├── evaluate.py               # 评估模块
├── visualize.py              # 可视化模块
└── utils.py                  # 工具函数
```

---

## 2. 数据准备与格式说明

### 2.1 数据格式

TrAISformer 使用 pickle (.pkl) 格式存储数据。每个数据集文件包含一个列表，列表中的每个元素是一个字典，代表一条船舶轨迹：

```python
# 单条轨迹的数据结构
trajectory = {
    "mmsi": 123456789,        # 船舶MMSI标识符（9位数字）
    "traj": numpy.array([    # 轨迹数据 (N, 5)
        [lat, lon, sog, cog, timestamp],  # 时间点 1
        [lat, lon, sog, cog, timestamp],  # 时间点 2
        # ...
    ])
}
```

**轨迹特征说明**：

| 索引 | 特征 | 含义 | 取值范围 |
|------|------|------|----------|
| 0 | lat | 纬度 | 0.0 - 1.0（归一化） |
| 1 | lon | 经度 | 0.0 - 1.0（归一化） |
| 2 | sog | 速度 (Speed Over Ground) | 0.0 - 1.0（归一化） |
| 3 | cog | 航向 (Course Over Ground) | 0.0 - 1.0（归一化） |
| 4 | timestamp | Unix 时间戳 | 整数 |

### 2.2 查看数据内容

你可以使用项目提供的 `pkl_viewer.py` 脚本查看数据：

```bash
# 查看训练数据的前几条轨迹
python pkl_viewer.py
```

**预期输出**：
```
Loading data/ct_dma/ct_dma_train.pkl...
Number of trajectories: 50000

First trajectory:
  MMSI: 123456789
  Length: 150 points
  Time range: 1609459200 - 1609545600
  Lat range: [0.45, 0.52]
  Lon range: [0.32, 0.38]
  SOG range: [0.12, 0.85]
  COG range: [0.05, 0.95]
```

### 2.3 数据转换工具

如果你有原始的 AIS CSV 数据，可以使用项目提供的转换工具：

```bash
# 将 CSV 转换为 PKL 格式
python csv_to_pkl_converter.py --input data/raw/ais_data.csv --output data/ct_dma/ct_dma_train.pkl

# 清洗异常值
python remove_outliers.py --input data/ct_dma/ct_dma_train.pkl --output data/ct_dma/ct_dma_train_clean.pkl
```

### 2.4 数据统计信息

在训练之前，了解数据分布很重要：

```bash
# 查看数据集基本信息（修改 pkl_viewer.py 添加统计）
python -c "
import pickle
with open('data/ct_dma/ct_dma_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Total trajectories: {len(data)}')
lengths = [len(d[\"traj\"]) for d in data]
print(f'Avg length: {sum(lengths)/len(lengths):.1f}')
print(f'Min length: {min(lengths)}')
print(f'Max length: {max(lengths)}')
"
```

---

## 3. 第一次训练：基础流程

### 3.1 配置参数

在开始训练之前，让我们先了解关键配置参数。打开 `config_transformer.py`：

```python
# 设备配置
device = torch.device("cuda:0")  # 使用 GPU

# 训练参数
max_epochs = 15          # 训练轮数
batch_size = 32          # 批大小
learning_rate = 6e-4     # 学习率

# 序列长度
init_seqlen = 18         # 输入序列长度（用于预测）
prediction_steps = 24    # 预测步数（24步 ≈ 4小时）
max_seqlen = 120         # 最大序列长度
```

### 3.2 启动训练

```bash
# 运行完整训练流程
python main.py
```

**预期输出**：
```
2024-01-15 10:30:00 - INFO - === TrAISformer Pipeline Started ===
2024-01-15 10:30:00 - INFO - Device: cuda:0
2024-01-15 10:30:00 - INFO - Dataset: ct_dma
2024-01-15 10:30:00 - INFO - Save directory: results/ct_dma_20240115_103000

Loading data/ct_dma/ct_dma_train.pkl...
Loading: 100%|██████████| 50000/50000 [00:02<00:00, 23456.87it/s]
Original: 50000, Filtered: 48500

Loading data/ct_dma/ct_dma_valid.pkl...
Original: 5000, Filtered: 4800

Loading data/ct_dma/ct_dma_test.pkl...
Original: 5000, Filtered: 4750

Creating pytorch dataset...
Creating pytorch dataset...
Creating pytorch dataset...

Model created with 1.24e+08 parameters

Starting training...

Epoch 1/15:
  Train Loss: 3.4521 | Valid Loss: 3.2845
  [Train] 100%|██████████| 1515/1515 [05:23<00:00,  4.68it/s]
  [Valid] 100%|██████████| 150/150 [00:45<00:00,  3.33it/s]
  
  Sampling trajectories for visualization...
  Saved visualization to results/ct_dma_20240115_103000/samples/epoch_1.png
```

### 3.3 训练过程解读

训练过程中你会看到以下信息：

1. **数据加载**：显示过滤后的有效样本数量
2. **模型参数量**：约 1.24 亿参数
3. **训练损失**：每个 epoch 显示训练和验证损失
4. **可视化样本**：每轮生成预测轨迹图

### 3.4 查看训练结果

训练完成后，结果保存在 `results/` 目录下：

```bash
# 查看结果目录
ls -la results/ct_dma_*/
```

**预期输出**：
```
results/ct_dma_20240115_103000/
├── log.txt                      # 训练日志
├── config.txt                   # 配置参数
├── best_model.pth              # 最佳模型权重
├── final_model.pth             # 最终模型权重
├── samples/
│   ├── epoch_1.png             # 每轮的预测样本
│   ├── epoch_2.png
│   └── ...
└── training_curves.png         # 训练曲线
```

---

## 4. 结果分析与可视化

### 4.1 训练曲线分析

训练完成后，查看 `training_curves.png` 了解训练过程：

- **损失曲线**：应该平稳下降，验证损失不应明显高于训练损失
- **过拟合迹象**：如果验证损失开始上升而训练损失继续下降，说明过拟合

### 4.2 预测轨迹可视化

每轮训练会生成预测轨迹对比图：

```python
# 图表说明：
# 蓝色实线：初始观测序列
# 绿色实线：真实未来轨迹
# 红色虚线：模型预测轨迹
```

**如何解读可视化图**：
- 预测轨迹越接近真实轨迹，模型效果越好
- 预测初期（1-2小时）误差应较小
- 随着预测时间增加，误差会逐渐累积

### 4.3 运行评估

训练完成后，可以单独运行评估：

```bash
# 仅运行评估
python main.py --eval-only
```

**预期输出**：
```
2024-01-15 14:30:00 - INFO - Loading model from results/ct_dma_20240115_103000/best_model.pth

Starting model evaluation...
Evaluating: 100%|██████████| 149/149 [00:32<00:00,  4.65it/s]

Prediction Error Statistics:
  Mean Error: 3.24 km
  Min Error: 0.85 km
  Max Error: 15.67 km
  
Error by Prediction Horizon:
  1 hour:  1.23 km
  2 hours: 2.45 km
  3 hours: 4.12 km
  4 hours: 5.89 km
```

### 4.4 生成可视化报告

```bash
# 仅运行可视化
python main.py --viz-only
```

这将生成误差曲线图和详细的统计报告。

---

## 5. 实际实验操作

### 实验一：调整预测时间范围

**目标**：将预测时间从 4 小时扩展到 6 小时

**操作步骤**：

1. 打开 `config_transformer.py`
2. 修改参数：

```python
# 修改预测步数
prediction_steps = 36   # 原来 24 → 36（约6小时）
prediction_hours = 6   # 原来 4 → 6
```

3. 重新训练：

```bash
python main.py --train-only
```

4. 观察结果变化

**预期变化**：
- 预测误差会随时间增加
- 6 小时预测误差通常比 4 小时高 30-50%

### 实验二：使用更长的观测序列

**目标**：增加初始观测长度，提高预测准确性

**操作步骤**：

1. 修改配置：

```python
init_seqlen = 24     # 原来 18 → 24（更长的历史数据）
```

2. 重新训练并比较结果

**预期结果**：
- 更长的观测序列通常能提高预测准确性
- 但会增加计算成本

### 实验三：调整模型规模

**目标**：实验不同规模的模型

**小模型配置**：

```python
n_layer = 4           # 原来 8 → 4
n_head = 4           # 原来 8 → 4
# 嵌入维度会自动调整
```

**大模型配置**：

```python
n_layer = 12         # 增加层数
n_head = 12          # 增加头数
```

**实验记录表**：

| 模型规模 | 参数量 | 训练时间/轮 | 预测误差 (4h) |
|----------|--------|-------------|---------------|
| 小 (4层4头) | ~30M | ~3分钟 | 6.5 km |
| 中 (8层8头) | ~124M | ~6分钟 | 5.2 km |
| 大 (12层12头) | ~280M | ~12分钟 | 4.8 km |

### 实验四：调整采样策略

**目标**：探索不同采样参数对预测的影响

**修改采样配置**：

```python
# 方案1：更确定的预测
sample_mode = "pos"      # 去掉邻近约束
top_k = 5                # 减少候选数量

# 方案2：更多样化的预测
sample_mode = "pos_vicinity"
r_vicinity = 60          # 扩大邻近范围
temperature = 1.2       # 增加随机性
```

---

## 6. 进阶实验与调优

### 6.1 学习率调度实验

学习率是影响训练的关键参数：

```python
# 实验不同的学习率
learning_rate = 3e-4    # 较低学习率
learning_rate = 1e-3    # 较高学习率
```

**学习率选择建议**：
- 如果损失不下降：尝试大学习率
- 如果损失震荡：尝试小学习率
- 从 6e-4 开始，根据结果调整

### 6.2 批量大小实验

批量大小影响训练稳定性和内存使用：

```python
# 根据 GPU 内存调整
batch_size = 16   # 小批量，内存占用低
batch_size = 32   # 中等批量（默认）
batch_size = 64   # 大批量，需要更多显存
```

**经验法则**：
- GPU 内存 8GB：batch_size = 16-32
- GPU 内存 16GB：batch_size = 32-64
- 批量越大，训练越稳定，但需要更多显存

### 6.3 数据增强

虽然项目未包含复杂的数据增强，但你可以尝试：

```python
# 在 datasets.py 的 __getitem__ 中添加

# 1. 随机噪声（模拟定位误差）
if augment and random.random() > 0.5:
    noise = torch.randn_like(seq) * 0.01
    seq = seq + noise

# 2. 随机时间掩码
if augment and random.random() > 0.5:
    mask_start = random.randint(0, seqlen // 2)
    seq[mask_start:] = 0
```

### 6.4 使用 TensorBoard 监控

如果想更详细地监控训练过程：

```python
# 在 config_transformer.py 中启用
tb_log = True
```

```bash
# 启动 TensorBoard
tensorboard --logdir results/
```

然后在浏览器中访问 `http://localhost:6006` 查看：
- 损失曲线
- 学习率变化
- 梯度统计
- 样本预测

---

## 7. 常见问题与解决方案

### 问题一：CUDA 内存不足

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

1. 减小批量大小：
```python
batch_size = 16
```

2. 减小序列长度：
```python
max_seqlen = 96
```

3. 减小模型规模：
```python
n_layer = 6
n_head = 6
```

4. 使用 CPU（临时方案）：
```python
device = torch.device("cpu")
```

### 问题二：训练损失不下降

**可能原因**：
- 学习率不合适
- 数据加载有问题
- 模型配置错误

**排查步骤**：

1. 检查数据是否正确加载：
```python
# 在 data_loader.py 中添加调试信息
print(f"Sample shape: {seqs.shape}")
print(f"Sample values: {seqs[0]}")
```

2. 检查损失值是否合理：
```python
# 初始损失应该约为 log(类别数)
# 对于 250 个类别，约等于 5.5
```

3. 尝试不同的学习率

### 问题三：预测轨迹不连续

**表现**：
- 预测的轨迹出现跳跃
- 位置变化不合理

**解决方案**：

1. 增加邻近采样范围：
```python
r_vicinity = 60
```

2. 降低采样温度：
```python
temperature = 0.8
```

3. 使用贪心解码：
```python
# 在 sample() 函数中设置
sample = False  # 使用概率最高的预测
```

### 问题四：过拟合

**表现**：
- 训练损失持续下降
- 验证损失开始上升

**解决方案**：

1. 增加正则化：
```python
weight_decay = 0.2
```

2. 使用 Dropout：
```python
attn_pdrop = 0.2
resid_pdrop = 0.2
```

3. 减少模型规模：
```python
n_layer = 6
```

---

## 8. 性能优化技巧

### 8.1 训练加速

```bash
# 使用多 GPU 训练（需要修改代码）
python -m torch.distributed.launch --nproc_per_node=4 main.py
```

```python
# 在配置中添加
use_amp = True  # 混合精度训练
```

### 8.2 推理加速

```python
# 在评估时使用
with torch.no_grad():
    preds = model(seqs)
```

```python
# 减少采样次数
n_samples = 8  # 从 16 减少到 8
```

### 8.3 内存优化

```python
# 训练后清理缓存
torch.cuda.empty_cache()

# 删除不需要的变量
del unused_variable
gc.collect()
```

---

## 9. 实践作业

完成以下作业来巩固所学知识：

### 作业一：基础训练（必做）

1. 启动训练并等待完成
2. 记录训练时间和最终损失
3. 分析预测误差随时间的变化

### 作业二：参数对比（必做）

1. 使用默认参数训练 baseline
2. 将 `prediction_steps` 改为 12，重新训练
3. 比较两次实验的误差曲线

### 作业三：可视化分析（选做）

1. 编写脚本加载训练好的模型
2. 选择测试集中的几条轨迹
3. 可视化预测结果并分析误差来源

### 作业四：模型改进（挑战）

1. 尝试修改模型架构（如添加残差连接）
2. 或尝试添加数据增强
3. 比较改进前后的效果

---

## 10. 总结与下一步

通过本教程，你应该已经：

- ✅ 掌握了环境配置和项目结构
- ✅ 理解了数据格式和预处理流程
- ✅ 能够运行完整的训练-评估流程
- ✅ 学会分析和可视化结果
- ✅ 掌握常见问题的调试方法
- ✅ 能够进行基本的实验设计

### 进一步学习建议

1. **深入模型原理**：阅读 Transformer 论文 "Attention Is All You Need"
2. **探索更多数据**：尝试其他 AIS 数据集
3. **阅读源码**：深入理解 `models.py` 中的模型实现
4. **参与社区**：与其他研究者交流经验和想法

如果你遇到任何问题或有新的发现，欢迎随时交流！

---

*本教程会持续更新，如有建议请提交 Issue 或 Pull Request。*
