import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

from config_transformer import Config
TB_LOG = Config.tb_log

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    给定一段条件序列（AIS 观测）seq，预测下一时刻的观测；
    每次预测都会被拼接回序列中，如此循环，共进行 steps 次（连续多步自回归采样）。
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # 如有需要，裁剪上下文长度

        # logits 形状: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # 取最后一个时间步的 logits，并按 temperature 进行缩放
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # 对每个通道做 softmax，得到类别概率
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # 从分布中采样（sample=True），或取概率最大的类别（sample=False）
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # 将离散类别索引还原为归一化坐标值 x（范围 [0,1)），0.5 相当于取对应 bin 的中心
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # 将新预测的一个时间步拼接回序列，进入下一轮
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


class TrainerConfig:
    # 优化相关参数（optimization parameters）
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # 仅对矩阵乘权重生效（matmul weights）
    # 学习率衰减参数：先线性 warmup，随后余弦退火至初始学习率的 10%
    lr_decay = False
    warmup_tokens = 375e6  # 参考 GPT-3 论文的设置，未必对所有任务都是最佳
    final_tokens = 260e9  # 达到该 token 数时学习率降到原始的 10%
    # 检查点（checkpoint）设置
    ckpt_path = None
    num_workers = 0  # DataLoader 子进程个数

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=getattr(config, 'pin_memory', False),
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # 将数据放到正确的设备上
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # 前向计算
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    loss = loss.mean()  # 若使用多卡，将各卡上的 loss 做均值
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # 反向传播并更新参数
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # 根据训练进度衰减学习率
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # 本步处理的 token 数
                        if self.tokens < config.warmup_tokens:
                            # 线性 warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # 余弦退火学习率
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # 进度展示
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # TensorBoard 日志
                    if TB_LOG:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch * n_batches + it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # 基于验证集 loss 的 early stopping；如果没有验证集则始终保存
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## 抽样与绘图（可视化预测）
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)
