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
INPUT_DIM = 15  # NdotL, NdotV, NdotH, VdotL, roughness, metallic, specular, thick, ao, curv, wrap, trans, fwd_scatter, thin_backlight, fresnel

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
    
    # 加载材质参数 (SSS-GGX-MLP: 每张渲染可能有不同的 roughness/metallic/specular)
    mat_path = os.path.join(data_dir, f'matparams_{idx:04d}.npy')
    if os.path.exists(mat_path):
        mat_params = np.load(mat_path)  # [roughness, metallic, specular]
        render_roughness, render_metallic, render_specular = mat_params
    else:
        render_roughness, render_metallic, render_specular = 0.4, 0.0, 0.5  # 默认值
    
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
        NdotL, NdotV, NdotH, VdotL,
        np.full_like(NdotL, render_roughness),
        np.full_like(NdotL, render_metallic),
        np.full_like(NdotL, render_specular),
        thick, ao, curv,
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

# [艺术夸张策略]: 物理上厚物体的背光 SSS 本来就很暗。
# 我们可以人为提亮训练目标中的“背光面 (NdotL < 0)”颜色的亮度。
# 当网络去学习这部分时，自然会给出更强烈的透射感。
# X[:, 0] 是 NdotL。
# max_boost=2.0 意味着在完全背光时 (-NdotL=1)，网络的目标被放大了 1+2.0=3.0 倍。
artificial_sss_boost = 1.0 
sss_enhance = 1.0 + artificial_sss_boost * np.maximum(-X[:, 0], 0.0)
Y_norm = Y_norm * sss_enhance[:, None]
print(f"已应用背光艺术增强 (Artifical SSS Boost) 提升透光感！最大增强倍数: {1.0+artificial_sss_boost}x")

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
    # 使用 Log-Space (对数空间) 进行监督，解决直接受光面压制暗部 SSS 细节的问题。
    # 为了避免预测值为负数导致 log1p 丢出 NaN 爆炸，需要对 pred 进行保护。
    # 我们用 torch.clamp 保护对数域输入，由于下方有线性的差值约束 (diff_lin)，
    # 负预测依然会吃到拉升向上的梯度，绝不会被困住死锁。
    scale = 10.0 # 增强暗部梯度区分度
    pred_safe = torch.clamp(pred, min=0.0)
    
    pred_log = torch.log1p(pred_safe * scale)
    target_log = torch.log1p(target * scale)
    
    diff_log = pred_log - target_log
    diff_lin = pred - target
    
    # 1. 对数 L1：核心误差，使网络平等对待暗部与亮部的相对差异
    l1_log = torch.mean(torch.abs(diff_log) * sss_channel_weight)
    
    # 2. 对数 MSE：惩罚巨大的相对误差（如网络没预测出应该有的暗部高亮）
    mse_log = torch.mean(diff_log ** 2 * sss_channel_weight)
    
    # 3. 线性 L1：保留少量的线性误差，保证在极端高光（且保护最初可能为负时）能持续提供梯度
    l1_lin = torch.mean(torch.abs(diff_lin) * sss_channel_weight) * 0.2
    
    return l1_log + mse_log + l1_lin


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
