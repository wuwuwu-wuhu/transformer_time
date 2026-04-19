# TrAISformer 训练流程与性能说明

## 1. 训练流水线概览
- **入口脚本**：`main.py` 将流程拆分为数据加载、训练、评估与可视化四个阶段，可通过命令行开关灵活组合。
- **配置管理**：`config_transformer.py` 提供模型与训练参数的集中配置，保证实验可复现。
- **执行结果**：训练与评估产物保存在 `results/` 下的自动命名目录，便于对比不同实验。

```mermaid
documentation
  title TrAISformer Pipeline
  section main.py
    数据加载
    模型训练
    模型评估
    结果可视化
  section train.py
    创建模型
    配置优化器
    调用 trainers.Trainer
  section trainers.py
    迭代批次
    调整学习率
    保存最佳权重
```

## 2. 关键节点详解
- **数据加载 (`data_loader.py`)**
  - 读取 `data/ct_dma/*.pkl`，移除低速/静止片段，加速收敛。
  - 使用 `datasets.AISDataset` 截断或补齐序列至 `max_seqlen+1`，生成批内对齐的 `seqs` 与 `mask`。
  - 根据阶段设置 `shuffle`，并令 `config.final_tokens = 2 * len(train) * max_seqlen` 以驱动余弦学习率衰减。

- **主流程 (`main.py`)**
  - 固定随机种子并记录设备信息。
  - 自动根据命令行切换阶段：默认执行完整流水线，可通过 `--train-only`、`--eval-only` 等参数定制。
  - 训练阶段调用 `train.run_training_pipeline()`，评估阶段调用 `evaluate.run_evaluation_pipeline()` 并复用最佳模型。

- **训练封装 (`train.py`)**
  - `create_model()` 构造 `models.TrAISformer`，记录参数量。
  - `run_training_pipeline()` 负责日志目录、TensorBoard（可选）、模型训练及权重加载。

- **训练循环 (`trainers.py`)**

```python
# trainers.py
for epoch in range(config.max_epochs):
    run_epoch('Training', epoch)
    test_loss = run_epoch('Valid', epoch)
    if good_model:
        best_loss = test_loss
        self.save_checkpoint(best_epoch + 1)
    preds = sample(raw_model, seqs_init, 96 - init_seqlen, sample=True)
```

  - `run_epoch()` 遍历 `DataLoader`，在 GPU 上执行前向与反向传播，梯度裁剪维持稳定。
  - 支持余弦退火学习率（`lr_decay=True` 时生效），使用 `AdamW` 优化器搭配权重衰减。
  - 每轮末尾自动调用 `sample()` 生成预测轨迹 JPG，提供快速质量反馈。

- **模型结构 (`models.py`)**
  - 量化纬度、经度、速度、航向为离散类别，分别嵌入后拼接，减少回归难度。
  - Encoder-Decoder 堆叠 `n_layer` 层，自注意力 + 交叉注意力捕捉长程依赖。
  - 输出头按纬度/经度/速度/航向拆分 softmax，多任务联合训练。

## 3. 为什么训练效率高？
- **高效数据预处理**：`load_ais_data()` 先剔除静止段与异常轨迹，再按最大长度截断，显著降低无效时间步。
- **GPU 友好设计**：默认使用 `Config.device = torch.device("cuda:0")`，所有张量在加载后立即迁移到 GPU，避免跨设备拷贝。
- **离散化输出空间**：模型预测离散类别而非连续数值，softmax 分类在 GPU 上高度优化，收敛速度快。
- **自适应学习率**：`Trainer` 通过 warmup + 余弦衰减在高学习率与稳定性间取得平衡，迭代次数更少。
- **梯度裁剪与正则化**：`torch.nn.utils.clip_grad_norm_` 与 `AdamW` 权重衰减抑制梯度爆炸，使得可以安全地使用更大的批量与学习率。
- **多线程数据加载**：`config.num_workers = 4`（可配置），在 I/O 与计算间并行，提高 GPU 利用率。
- **即时可视化反馈**：每轮生成预测轨迹图，有助于快速发现异常并调整实验，减少无效训练。

## 4. 训练相关文件说明
- **`config_transformer.py`**：集中管理模型规模、数据路径、模糊采样设置、学习率调度及保存目录命名。
- **`main.py`**：命令行入口，统筹数据、训练、评估、可视化阶段。
- **`data_loader.py`**：从 PKL 构造 `AISDataset`/`AISDataset_grad`，并创建 `DataLoader`。
- **`datasets.py`**：定义数据集张量化逻辑（归一化、padding、掩码）。
- **`models.py`**：`TrAISformer` 主体，包含嵌入层、编码器/解码器块、自定义采样函数等。
- **`train.py`**：负责搭建训练流水线、创建模型、处理日志与断点。
- **`trainers.py`**：实现训练循环、梯度更新、学习率调度、可视化采样与权重保存。
- **`utils.py`**：工具函数，包括随机种子设置、日志初始化、top-k 筛选等。
- **`evaluate.py`**（可选了解）：加载最佳模型执行评估，生成误差统计及可视化所需数据。

## 5. 快速上手示例
```bash
# 运行完整训练-评估-可视化
python main.py

# 仅训练并保存模型
python main.py --train-only

# 指定 GPU 并调整预测步长
python main.py --train-only \
  && python main.py --eval-only
```

如需自定义实验，修改 `config_transformer.py` 中如 `prediction_steps`、`n_layer`、`learning_rate` 等参数，然后重新运行训练阶段即可。
