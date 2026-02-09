import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from MAADS import ViViT
from MAADS import Transformer
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils import set_params_lr
from aviframe import RGBDDataset, custom_collate
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import sys
from focalloss import FocalLoss
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', required=True)
parser.add_argument('--mode', default='rgb')
parser.add_argument('-d','--device', default='cuda:1')
args = parser.parse_args()

mode = args.mode
frames = 80
num_classes = 6
device = args.device if torch.cuda.is_available() else 'cpu'

# モデル定義
model = ViViT(num_classes=num_classes, num_frames=frames, target_pruning_rate=0.5)
model = model.to(device)

# データセットとローダー
train_dataset = RGBDDataset('/home/higashi/symlink_hdd/higashi/dataset/train.txt', total_frames=frames, type=mode, device=device)
test_dataset = RGBDDataset('/home/higashi/symlink_hdd/higashi/dataset/test.txt', total_frames=frames, type=mode, device=device)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
pre_process = lambda data, device: list(map(lambda x: x.to(device), data))

# 重み計算
label_count = dict(Counter(train_loader.dataset.labels))
total_samples = len(train_loader.dataset.labels)
weights = torch.tensor([(total_samples / label_count[i]) / num_classes for i in range(num_classes)]).to(device)
criterion = nn.CrossEntropyLoss(weights)

# オプティマイザと学習率
lr = 1e-4
params = set_params_lr(model, module_names=model.module_names, backbone_name='space_transformer', lr=lr, backbone_lr_multiplier=0.1)
optimizer = optim.Adam(params, lr=lr)

# 訓練用変数
model_name = args.name
num_epochs = 100
best_acc = 0.
train_losses = []
test_accuracies = []
prune_rates = []
flop_reductions = []

# 初期FLOPsとParamの取得
sample_inputs, _ = next(iter(test_loader))
sample_inputs = pre_process(sample_inputs, device)
fca = FlopCountAnalysis(model, tuple(sample_inputs))
base_flops = fca.total()
base_params = sum(p.numel() for p in model.parameters())
param_table = parameter_count_table(model)
print("Base Model Param Table:\n", param_table)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_loss_threshold = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = pre_process(inputs, device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        
        loss_threshold = model.get_threshold_loss()
        loss += loss_threshold
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_threshold += loss_threshold.item()
        # if i == 2: break

    model.eval()
    correct = 0
    total = 0
    prune_sum = 0
    prune_count = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = pre_process(inputs, device), labels.to(device)
            outputs = model(*inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if hasattr(model, 'frame_selector'):
                with torch.no_grad():
                    # rgb_cls, depth_cls, skeleton_cls = model.frame_selector(*[x[:, :, 0] for x in inputs])
                    # weights = model.frame_selector(rgb_cls, depth_cls, skeleton_cls).squeeze(-1)
                    # prune_rate = (weights < torch.sigmoid(model.learnable_threshold)).float().mean().item()
                    prune_rate = model.prune_info['topk'] / model.prune_info['frame']
                    prune_sum += 1 - prune_rate
                    prune_count += 1
            # if prune_count == 2: break  # デバッグ用に2サンプルのみ
    running_loss /= len(train_loader)
    running_loss_threshold /= len(train_loader)
    acc = correct / total
    avg_prune_rate = prune_sum / prune_count if prune_count else 0
    fca = FlopCountAnalysis(model, tuple(sample_inputs))
    current_flops = fca.total()
    flop_reduction = 100 * (base_flops - current_flops) / base_flops

    train_losses.append(running_loss)
    test_accuracies.append(acc)
    prune_rates.append(avg_prune_rate)
    flop_reductions.append(flop_reduction)
    print(f"Epoch {epoch+1}, Loss_total {running_loss:.4f}, Loss_thresh {running_loss_threshold:.4f}, Acc: {acc*100:.2f}%, Prune Rate: {avg_prune_rate*100:.2f}%, FLOP ↓: {flop_reduction:.2f}%")

    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
        torch.save(model.state_dict(), f'{model_name}_best.pth')

# 結果の保存
torch.save(model.state_dict(), f'{model_name}_last.pth')
with open(f'results_{model_name}.txt', 'w') as f:
    f.write(f'Best Acc (Epoch {best_epoch+1}): {best_acc:.3f}\n')
    f.write(f'Final Acc: {acc:.3f}\n')
    f.write(f'Mean Prune Rate: {np.mean(prune_rates)*100:.2f}%\n')
    f.write(f'Mean FLOP Reduction: {np.mean(flop_reductions):.2f}%\n')

np.savez(f'./result_temp_{model_name}',
         train_losses=train_losses,
         test_accuracies=test_accuracies,
         prune_rates=prune_rates,
         flop_reductions=flop_reductions)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_accuracies, label='Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'performance_{model_name}.png')
