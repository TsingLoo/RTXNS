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
# 1. 网络结构（针对纯 Diffuse SSS 优化的配置）
# =========================================
class NeuralSSS(nn.Module):
    def __init__(self):
        super().__init__()
        width = 256  # increased capacity
        self.layers = nn.Sequential(
            nn.Linear(6, width), # Raw 6 angular/geometric inputs
            nn.Softplus(),
            nn.Linear(width, width),
            nn.Softplus(),
            nn.Linear(width, width),
            nn.Softplus(),
            nn.Linear(width, 3),
        )

    def forward(self, x):
        return self.layers(x)  # Raw output; max pool at inference to avoid negative values


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
    alpha_mask = render_rgba[:, :, 3] >= 0.5
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
    
    inputs_batch = np.stack([
        NdotL, NdotV, VdotL, thick, ao, curv
    ], axis=-1)
    
    all_inputs.append(inputs_batch)
    all_targets.append(rgb)

print(f"\n拼接完成！")
X = np.concatenate(all_inputs, axis=0).astype(np.float32)
Y = np.concatenate(all_targets, axis=0).astype(np.float32)
print(f"总计提取 {len(X)} 个有效黄金训练像素。")

# 颜色解耦 (Demodulation): 将绝对渲染 RGB 转换为相对的透射率 Transmittance
# 这样神经网络只学习光分布，不死记硬背物体本身的颜色！
base_color = np.array([0.28, 0.58, 0.22], dtype=np.float32)
Y_norm = Y / (base_color + 1e-6) # 我们不再做任何压缩！直接让网络爆破学习纯正无限 HDR！

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
# 2.5 准备无阻塞实时预览 (提取一张视角作为监视器)
# =========================================
print("准备实时预览窗口...")
def load_exr_channels(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img[:, :, [2, 1, 0, 3]] if img.shape[2] == 4 else img[:, :, [2, 1, 0]]

def safe_load_exr(file_base):
    path = os.path.join(data_dir, f'{file_base}_0000.exr')
    if not os.path.exists(path): path = os.path.join(data_dir, f'{file_base}.exr')
    return load_exr_channels(path)

try:
    test_rgba = load_exr_channels(os.path.join(data_dir, 'render_0000.exr'))
    test_light = np.load(os.path.join(data_dir, 'lightdir_0000.npy'))
    test_L = test_light / (np.linalg.norm(test_light) + 1e-8)
    
    pos_raw = safe_load_exr('position_map')[:,:,:3]
    norm_raw = safe_load_exr('normal_map')[:,:,:3]
    
    center = np.load(os.path.join(data_dir, 'bbox_center.npy'))
    size = np.load(os.path.join(data_dir, 'bbox_size.npy'))[0]
    pos_map = (pos_raw - 0.5) * size + center
    norm_map = (norm_raw - 0.5) * 2.0
    thick_map = safe_load_exr('thickness_map')
    thick_map = thick_map[:,:,0] if thick_map.ndim == 3 else thick_map
    ao_map = safe_load_exr('ao_map')
    ao_map = ao_map[:,:,0] if ao_map.ndim == 3 else ao_map
    curv_map = safe_load_exr('curvature_map')
    curv_map = curv_map[:,:,0] if curv_map.ndim == 3 else curv_map
    
    cam_file = os.path.join(data_dir, 'camera_pos.npy')
    test_cam = np.load(cam_file) if os.path.exists(cam_file) else np.array([0.0, -5.0, 1.0])
    
    test_h, test_w = test_rgba.shape[:2]
    preview_pixels = []
    preview_features = []
    test_gt_img = np.zeros((test_h, test_w, 3), dtype=np.float32)
    
    for y in range(test_h):
        for x in range(test_w):
            test_gt_img[y, x] = test_rgba[y, x, :3]
            if test_rgba[y, x, 3] < 0.5: continue
            
            V = test_cam - pos_map[y, x]
            V_len = np.linalg.norm(V)
            if V_len < 1e-8: continue
            V = V / V_len
            
            n = norm_map[y, x]
            n_len = np.linalg.norm(n)
            if n_len < 1e-8: continue
            n = n / n_len
            
            preview_features.append([
                np.dot(n, test_L), max(np.dot(n, V), 0.0), np.dot(V, test_L),
                thick_map[y, x], ao_map[y, x], curv_map[y, x]
            ])
            preview_pixels.append((y, x))
            
    test_X_tensor = torch.tensor(preview_features, dtype=torch.float32).cuda()
    
    def simple_aces(x):
        x = np.maximum(0, x - 0.004)
        return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

    plt.ion() # 开启无阻塞模式
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Live Training Preview')
    im_pred = axes[0].imshow(np.zeros((test_h, test_w, 3)))
    axes[0].set_title("MLP Epoch 0")
    axes[0].axis("off")
    axes[1].imshow(np.clip(simple_aces(test_gt_img), 0, 1))
    axes[1].set_title("Blender Ground Truth")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

except Exception as e:
    print("预览初始化失败，跳过预览:", e)
    test_X_tensor = None


def log_mse_loss(pred, target):
    # 使用 L1-Relative Loss (近似 MAPE) 替代直接的 Log MSE
    # 这能完美解决 pred 为负数时的 NaN 导致网络崩溃的问题，
    # 同时保留对暗部的强力惩罚（当 target 很小时，分母很小，Loss 放大倍数极高）
    weight = 1.0 / (target + 0.05)
    return torch.mean(torch.abs(pred - target) * weight)

# =========================================
# 3. 训练
# =========================================
model = NeuralSSS().cuda()
epochs = 200
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
loss_fn = log_mse_loss

best_val_loss = float("inf")
patience = 25
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

    # ---- ----
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

    # ---- 实时渲染刷新 ----
    if test_X_tensor is not None:
        model.eval()
        with torch.no_grad():
            # 直接乘回 Base Color 完成渲染闭环！不再做任何逆向 Tone Mapping！
            preds_hdr = model(test_X_tensor).cpu().numpy()
            preds_hdr = np.maximum(preds_hdr, 0.0) # 防止轻微负数
            preds = preds_hdr * base_color
            temp_img = np.zeros((test_h, test_w, 3), dtype=np.float32)
            for i, (py, px) in enumerate(preview_pixels):
                temp_img[py, px] = preds[i]
            im_pred.set_data(np.clip(simple_aces(temp_img), 0, 1))
            axes[0].set_title(f"MLP Epoch {epoch+1}\nLoss: {val_loss:.4f}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

    # 保存最佳模型 + Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), f"{data_dir}/best_model.pt")
        print(f"  → 最佳模型已保存 (val_loss: {val_loss:.6f})")
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
# 2. 内存热转换：直接导出一份供 C++ RTXNS 引擎消费的 JSON
import json

out_data = {"layers": [], "input_dim": model.encoder.output_dim // (1 + model.encoder.freq_bands * 2), "freq_bands": model.encoder.freq_bands}
# 你的 nn.Sequential 中真实的 Linear 层的索引是 0, 2, 4, 6 (夹着 ReLU)
for i in [0, 2, 4, 6]:
    w = state_dict[f"layers.{i}.weight"].cpu().numpy()
    b = state_dict[f"layers.{i}.bias"].cpu().numpy()

    out_data["layers"].append(
        {
            "num_inputs": w.shape[1],
            "num_outputs": w.shape[0],
            "weights": w.flatten().tolist(),  # 碾平为 C++ 识别的一维数组
            "biases": b.flatten().tolist(),
        }
    )
# 导出训练时的归一化参数，C++ 推理端必须使用完全相同的值！
out_data["y_scale"] = [1.0, 1.0, 1.0]
out_data["base_color"] = base_color.tolist()
json_path = f"{data_dir}/jade_sss_weights.json"
with open(json_path, "w") as f:
    json.dump(out_data, f)
print(f"跨语言同步导出完成: {json_path} (供 C++ RTXNS 加载)")
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

    # 防止你在第三步忘了改 L1Loss，我们在这里加上 MSE 和 L1 的局部双对比提示
    for i in range(5):
        p = sample_pred[i]
        g = sample_gt[i]
        print(
            f"  样本{i}: 预测=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
            f"真实=({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f})"
        )
