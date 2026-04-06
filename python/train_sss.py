import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import save_file
import cv2
import matplotlib.pyplot as plt

# =========================================
# 1. 网络结构（残差 + SSS 特征）
# =========================================
INPUT_DIM = 12  # NdotL, NdotV, NdotH, VdotL, thick, ao, curv, wrap, trans, fwd_scatter, thin_backlight, fresnel

class NeuralSSS(nn.Module):
    def __init__(self):
        super().__init__()
        width = 64
        self.input_proj = nn.Linear(INPUT_DIM, width)
        self.act = nn.Softplus()
        self.block1 = nn.Sequential(nn.Linear(width, width), nn.Softplus())
        self.block2 = nn.Sequential(nn.Linear(width, width), nn.Softplus())
        self.block3 = nn.Sequential(nn.Linear(width, width), nn.Softplus())
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        h = self.act(self.input_proj(x))
        h = self.block1(h) + h  # residual skip
        h = self.block2(h) + h  # residual skip
        h = self.block3(h) + h  # residual skip
        return self.output(h)


# =========================================
# 2. 从原始 EXR 动态组装训练数据
# =========================================
print("正在从原始 EXR 图像动态解析物理特征和色彩...")
data_dir = "c:/tmp/jade_sss_data/"

import OpenEXR, Imath
def load_exr_channels_openexr(path, channels=('R', 'G', 'B')):
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    res = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(h, w) for c in channels]
    return np.stack(res, axis=-1)

def load_exr_single(path, channel='R'):
    return load_exr_channels_openexr(path, (channel,))[:,:,0]

thickness_map = load_exr_single(os.path.join(data_dir, 'thickness_map.exr'), 'R')
ao_map = load_exr_single(os.path.join(data_dir, 'ao_map.exr'), 'R')
normal_map = load_exr_channels_openexr(os.path.join(data_dir, 'normal_map.exr'))
position_map = load_exr_channels_openexr(os.path.join(data_dir, 'position_map.exr'))
cam_pos = np.load(os.path.join(data_dir, 'camera_pos.npy'))

# Load curvature if exists, else fallback to zeros
curv_path = os.path.join(data_dir, 'curvature_map.exr')
if os.path.exists(curv_path):
    curvature_map = load_exr_single(curv_path, 'R')
else:
    curvature_map = np.zeros_like(thickness_map)

all_inputs = []
all_targets = []
n_renders = len([f for f in os.listdir(data_dir) if f.startswith('render_') and f.endswith('.exr')])

import sys
for idx in range(n_renders):
    sys.stdout.write(f"\r处理渲染图 {idx+1}/{n_renders} ...")
    sys.stdout.flush()
    render_rgba = load_exr_channels_openexr(os.path.join(data_dir, f'render_{idx:04d}.exr'), ('R', 'G', 'B', 'A'))
    light_dir = np.load(os.path.join(data_dir, f'lightdir_{idx:04d}.npy'))
    L = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    
    h, w = render_rgba.shape[:2]
    
    # 向量化处理以极大加速拼接
    alpha_mask = render_rgba[:, :, 3] >= 0.99
    valid_y, valid_x = np.where(alpha_mask)
    
    rgb = render_rgba[valid_y, valid_x, :3]
    pos_raw = position_map[valid_y, valid_x]
    norm_raw = normal_map[valid_y, valid_x]
    thick = thickness_map[valid_y, valid_x]
    ao = ao_map[valid_y, valid_x]

    # 还原 Blender 发射器 Color clamping 之前的数据
    center = np.load(os.path.join(data_dir, 'bbox_center.npy'))
    size = np.load(os.path.join(data_dir, 'bbox_size.npy'))[0]
    pos = (pos_raw - 0.5) * size + center
    norm = (norm_raw - 0.5) * 2.0

    V = cam_pos - pos
    V_len = np.linalg.norm(V, axis=-1, keepdims=True)
    valid_v = (V_len[:, 0] >= 1e-8)
    
    norm_len = np.linalg.norm(norm, axis=-1, keepdims=True)
    valid_n = (norm_len[:, 0] >= 1e-8)
    
    valid_mask = valid_v & valid_n
    
    # 过滤无效法线或视向量
    rgb = rgb[valid_mask]
    pos = pos[valid_mask]
    norm = norm[valid_mask] / norm_len[valid_mask]
    thick = thick[valid_mask]
    ao = ao[valid_mask]
    curv = curvature_map[valid_y, valid_x][valid_mask]
    V = V[valid_mask] / V_len[valid_mask]
    
    NdotL = np.sum(norm * L, axis=-1)
    NdotV = np.maximum(np.sum(norm * V, axis=-1), 0.0)
    VdotL = np.sum(V * L, axis=-1)
    
    # SSS-GGX-MLP: 计算 Half Vector 和 NdotH，让 MLP 学习 GGX 高光峰
    H = V + L  # broadcast: L is (3,), V is (N, 3)
    H_len = np.linalg.norm(H, axis=-1, keepdims=True)
    H = H / np.maximum(H_len, 1e-8)
    NdotH = np.maximum(np.sum(norm * H, axis=-1), 0.0)
    
    # ======== SSS 专属特征 ========
    wrap_lighting = np.clip(NdotL * 0.5 + 0.5, 0.0, 1.0)
    transmission = np.exp(-thick * 3.0) * np.maximum(-NdotL, 0.0)
    forward_scatter = np.power(np.maximum(VdotL, 0.0), 4.0)
    thin_backlight = (1.0 - thick) * np.maximum(-NdotL, 0.0)
    fresnel = np.power(1.0 - NdotV, 5.0)  # Schlick 菲涅尔，编码润泽边缘光
    
    inputs_batch = np.stack([
        NdotL, NdotV, NdotH, VdotL, thick, ao, curv,
        wrap_lighting, transmission, forward_scatter, thin_backlight, fresnel
    ], axis=-1)
    
    all_inputs.append(inputs_batch)
    all_targets.append(rgb)

print(f"\n拼接完成！")
X = np.concatenate(all_inputs, axis=0).astype(np.float32)
Y = np.concatenate(all_targets, axis=0).astype(np.float32)
print(f"总计提取 {len(X)} 个有效黄金训练像素。")

# 光强归一化: 除以 Blender Sun energy，使 MLP 学习"单位光强下的 SSS 响应"
# 渲染方程对光源辐照度是线性的: render(E) = E × render(1)
# 所以 render(1) = render(E) / E
sun_energy_file = os.path.join(data_dir, 'sun_energy.npy')
if os.path.exists(sun_energy_file):
    sun_energy = np.load(sun_energy_file)[0]
    Y = Y / sun_energy
    print(f"已除以 Sun energy = {sun_energy}，归一化为单位光强响应")
else:
    print("警告: 未找到 sun_energy.npy，跳过光强归一化（旧数据兼容模式）")

# 颜色解耦: 网络学习 transmittance = RGB / base_color
# 这样换色时只需替换 base_color，不用重新训练
# SSS 的波长差异 (R/G/B 散射半径不同) 在 transmittance 中自然保留
base_color = np.array([0.28, 0.58, 0.22], dtype=np.float32)
Y_norm = Y / (base_color + 1e-6)

# 90/10 划分 train/val
n = len(X)
perm = np.random.RandomState(42).permutation(n)
split = int(n * 0.9)

X_train, X_val = X[perm[:split]], X[perm[split:]]
Y_train, Y_val = Y_norm[perm[:split]], Y_norm[perm[split:]]

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

train_loader = DataLoader(
    train_dataset, batch_size=65536, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=65536, shuffle=False, num_workers=0, pin_memory=True
)

print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")

# =========================================
# 2.5 启动独立预览进程
# =========================================
import subprocess
import atexit

print("启动独立实时预览进程 (evaluate_render.py)...")
try:
    preview_proc = subprocess.Popen(["python", "evaluate_render.py"])
    
    # 确保主进程退出时连带杀掉预览子进程
    @atexit.register
    def kill_preview():
        if preview_proc.poll() is None:
            preview_proc.terminate()
except Exception as e:
    print(f"警告：无法启动 evaluate_render.py 子进程: {e}")

# =========================================
# SSS-Aware Loss: 同时兼顾暗部精度和亮部通透感
# =========================================
# Blender SSS Radius (0.8, 1.2, 0.5) → 通道重要性权重
sss_radius = torch.tensor([0.8, 1.2, 0.5], device='cuda', dtype=torch.float32)
sss_channel_weight = sss_radius / sss_radius.max()  # [0.667, 1.0, 0.417]

def sss_aware_loss(pred, target):
    diff = pred - target
    
    # 1. L1: 线性空间，不做任何压缩，保留完整动态范围
    l1 = torch.mean(torch.abs(diff) * sss_channel_weight)
    
    # 2. MSE: 平方误差对大偏差（暗部偏亮 + 亮部偏暗）施加更强惩罚
    mse = torch.mean(diff ** 2 * sss_channel_weight)
    
    # 3. 对比度项：显式惩罚"该暗不暗"和"该亮不亮"
    #    暗部 (target < 0.02): pred 偏高时额外惩罚 (raw RGB scale)
    dark_penalty = torch.mean(torch.clamp(pred - 0.02, min=0.0) * (target < 0.02).float())
    #    亮部通透区 (target > 0.3): pred 偏低时额外惩罚  
    bright_penalty = torch.mean(torch.clamp(0.2 - pred, min=0.0) * (target > 0.3).float())
    
    return l1 + mse + 0.5 * (dark_penalty + bright_penalty)


import json
def export_model_json(model, path):
    """Export residual network to flat sequential format for C++ inference.
    
    The residual connections (h = block(h) + h) are mathematically equivalent to:
    - Adding an identity matrix to the weight matrix
    - Passing through the bias unchanged
    So we export the "effective" weights = W + I for hidden-to-hidden layers.
    """
    sd = model.state_dict()
    out_data = {"layers": [], "input_dim": INPUT_DIM, "freq_bands": 0}
    
    # Layer 0: input_proj (INPUT_DIM → 256)
    w0 = sd["input_proj.weight"].cpu().numpy()
    b0 = sd["input_proj.bias"].cpu().numpy()
    out_data["layers"].append({
        "num_inputs": w0.shape[1],
        "num_outputs": w0.shape[0],
        "weights": w0.flatten().tolist(),
        "biases": b0.flatten().tolist(),
    })
    
    # Layers 1-3: residual blocks → export RAW weights (不融合！)
    # 残差跳连必须在 shader 中显式实现：h = softplus(W*h + b) + h_saved
    # 因为 softplus(W*h + b) + h ≠ softplus((W+I)*h + b)  ← 非线性不可交换
    for block_idx in range(1, 4):
        w = sd[f"block{block_idx}.0.weight"].cpu().numpy()
        b = sd[f"block{block_idx}.0.bias"].cpu().numpy()
        out_data["layers"].append({
            "num_inputs": w.shape[1],
            "num_outputs": w.shape[0],
            "weights": w.flatten().tolist(),
            "biases": b.flatten().tolist(),
        })
    
    # Layer 4: output (256 → 3)
    w_out = sd["output.weight"].cpu().numpy()
    b_out = sd["output.bias"].cpu().numpy()
    out_data["layers"].append({
        "num_inputs": w_out.shape[1],
        "num_outputs": w_out.shape[0],
        "weights": w_out.flatten().tolist(),
        "biases": b_out.flatten().tolist(),
    })
    
    out_data["y_scale"] = [1.0, 1.0, 1.0]
    out_data["base_color"] = base_color.tolist()
    with open(path, "w") as f:
        json.dump(out_data, f)

# =========================================
# 3. 训练
# =========================================
model = NeuralSSS().cuda()
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
loss_fn = sss_aware_loss

best_val_loss = float("inf")
patience = 40
no_improve_count = 0

print("开始训练...")
for epoch in range(epochs):
    # ---- Train ----
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()
        pred = model(bx)
        loss = loss_fn(pred, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(bx)
        train_count += len(bx)

    # ---- Val ----
    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.cuda(), by.cuda()
            pred = model(bx)
            loss = loss_fn(pred, by)
            val_loss_sum += loss.item() * len(bx)
            val_count += len(bx)

    train_loss = train_loss_sum / train_count
    val_loss = val_loss_sum / val_count
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    print(
        f"Epoch {epoch+1:3d}/{epochs}  "
        f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  LR: {lr:.6f}"
    )

    # 保存最佳模型 + Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), f"{data_dir}/best_model.pt")
        export_model_json(model, f"{data_dir}/jade_sss_weights.json")
        print(f"  → 最佳模型已保存 (val_loss: {val_loss:.6f}) 并已同步导出 JSON!")
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

# =========================================
# 4. 导出 safetensors 与 JSON
# =========================================
print("\n加载最佳模型...")
model.load_state_dict(torch.load(f"{data_dir}/best_model.pt"))
model.eval()
state_dict = model.state_dict()
# 1. 导出供以后纯 Python 分析用的 safetensors
state_dict["y_scale"] = torch.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32))
save_file(state_dict, f"{data_dir}/jade_sss_weights.safetensors")
print(f"导出完成: {data_dir}/jade_sss_weights.safetensors")
# 2. 导出完毕
print(f"  → y_scale (ToneMapped) = [1.0, 1.0, 1.0]")
print(f"  → base_color = {base_color.tolist()}")
# =========================================
# 5. 快速质量检查
# =========================================
print("\n抽样验证:")
model.eval()
with torch.no_grad():
    sample_x = torch.from_numpy(X_val[:5]).cuda()
    sample_hdr = model(sample_x).cpu().numpy()
    sample_hdr = np.maximum(sample_hdr, 0.0)
    sample_pred = sample_hdr * base_color
    sample_gt = Y_val[:5] * base_color

    for i in range(5):
        p = sample_pred[i]
        g = sample_gt[i]
        print(
            f"  样本{i}: 预测=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
            f"真实=({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f})"
        )
