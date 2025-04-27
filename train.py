import torch
import torch.nn as nn
import time

from loss.label_smoothing import LabelSmoothing
from optim.schedulers import NoamOpt, get_std_opt
# from utils.multigpu_loss import MultiGPULossCompute

from model import make_model       # 假设你的模型构建函数写在 model.py
from dataloader import prepare_data      # 假设你的 prepare_data 函数写在 data.py

def SimpleLossCompute(generator, criterion, opt=None, device=None):
    def compute(x, y, norm):
        if device is not None:
            x = x.to(device)
            y = y.to(device)

        x = generator(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        y_flat = y.contiguous().view(-1)
        loss = criterion(x_flat, y_flat) / norm

        if opt is not None:
            loss.backward()
            opt.step()
            opt.optimizer.zero_grad()

        return loss.item() * norm
    return compute

def run_epoch(epoch, data_iter, model, loss_compute, device):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    model.train()
    for i, batch in enumerate(data_iter):
        batch = batch.to(device)  # 确保 batch 数据也转移到设备上

        out = model(batch.trg, batch.src, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"Epoch {epoch} | Step {i} | Loss {loss:.2f} | Tokens/sec {tokens / elapsed:.2f}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def main():
    # === 设置设备 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Hyperparameters for GPU===
    # batch_size = 64
    # N = 6
    # d_model = 512
    # d_ff = 2048
    # h = 8
    # dropout = 0.1
    # epochs = 10

    # === Hyperparameters for CPU=== 
    batch_size = 4         # 小 batch
    N = 2                  # 层数更小
    d_model = 128          # 维度降低
    d_ff = 512             # FFN 更小
    h = 2                  # 多头数量减少
    dropout = 0.1
    epochs = 5             # 只跑 1 epoch 验证流程是否通

    # === Data ===
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_data(batch_size=batch_size
                                                                               , small_data=True)

    # === Model ===
    model = make_model(len(src_vocab), len(tgt_vocab), N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout).to(device)

    # === Criterion ===
    padding_idx = tgt_vocab["<blank>"]
    criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=padding_idx, smoothing=0.1).to(device)
    
    # === Optimizer ===
    model_opt = get_std_opt(model)

    # === Loss Compute ===
    loss_compute = SimpleLossCompute(model.generator, criterion, model_opt, device)

    # === Training Loop ===
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(epoch, train_loader, model, loss_compute, device)
        print(f"Epoch {epoch} completed | Training loss: {train_loss:.4f}")

        # Evaluation（可选）
        with torch.no_grad():
            model.eval()
            val_loss = run_epoch(epoch, val_loader, model, SimpleLossCompute(model.generator, criterion, None, device), device)
            print(f">>> Validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()