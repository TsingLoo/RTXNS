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
class FrequencyEncoder(nn.Module):
    def __init__(self, input_dim=7, freq_bands=4): # increased freq_bands for better high frequency recovery
        super().__init__()
        self.freq_bands = freq_bands
        self.output_dim = input_dim * (1 + freq_bands * 2)

    def forward(self, x):
        encodings = [x]
        for freq in range(self.freq_bands):
            encodings.append(torch.sin((2.0**freq) * torch.pi * x))
            encodings.append(torch.cos((2.0**freq) * torch.pi * x))
        return torch.cat(encodings, dim=-1)

class NeuralSSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrequencyEncoder(input_dim=7, freq_bands=4)
        width = 128  # increased capacity
        self.layers = nn.Sequential(
            nn.Linear(self.encoder.output_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 3),
        )

    def forward(self, x):
        x = self.encoder(x)
        return torch.sigmoid(self.layers(x))


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
    pos = position_map[valid_y, valid_x]
    norm = normal_map[valid_y, valid_x]
    thick = thickness_map[valid_y, valid_x]
    ao = ao_map[valid_y, valid_x]

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
    V = V[valid_mask] / V_len[valid_mask]
    
    NdotL = np.sum(norm * L, axis=-1)
    NdotV = np.maximum(np.sum(norm * V, axis=-1), 0.0)
    VdotL = np.sum(V * L, axis=-1)
    
    inputs_batch = np.stack([
        NdotL, NdotV, VdotL, thick, ao,
        np.full_like(NdotL, 0.15), np.zeros_like(NdotL)
    ], axis=-1)
    
    all_inputs.append(inputs_batch)
    all_targets.append(rgb)

print(f"\n拼接完成！")
X = np.concatenate(all_inputs, axis=0).astype(np.float32)
Y = np.concatenate(all_targets, axis=0).astype(np.float32)
print(f"总计提取 {len(X)} 个有效黄金训练像素。")

Y = np.clip(Y, 0.0, 5.0)
# 归一化 target 到 [0, 1]（sigmoid 输出范围）
y_max = Y.max(axis=0, keepdims=True)
y_max = np.maximum(y_max, 1e-6)  # 防止除零
Y_norm = Y / y_max
print(f"Target 各通道最大值: R={y_max[0,0]:.4f} G={y_max[0,1]:.4f} B={y_max[0,2]:.4f}")

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
    
    pos_map = safe_load_exr('position_map')[:,:,:3]
    norm_map = safe_load_exr('normal_map')[:,:,:3]
    thick_map = safe_load_exr('thickness_map')
    thick_map = thick_map[:,:,0] if thick_map.ndim == 3 else thick_map
    ao_map = safe_load_exr('ao_map')
    ao_map = ao_map[:,:,0] if ao_map.ndim == 3 else ao_map
    
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
                thick_map[y, x], ao_map[y, x], 0.15, 0.0
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


# =========================================
# 3. 训练
# =========================================
model = NeuralSSS().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=80, eta_min=1e-5
)
# MSE 强迫网络去拟合差距巨大的高光，而不是像 L1 那样把它当做离群点抛弃
loss_fn = nn.MSELoss()

best_val_loss = float("inf")
epochs = 80

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
            preds = model(test_X_tensor).cpu().numpy() * y_max.flatten()
            temp_img = np.zeros((test_h, test_w, 3), dtype=np.float32)
            for i, (py, px) in enumerate(preview_pixels):
                temp_img[py, px] = preds[i]
            im_pred.set_data(np.clip(simple_aces(temp_img), 0, 1))
            axes[0].set_title(f"MLP Epoch {epoch+1}\nLoss: {val_loss:.4f}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{data_dir}/best_model.pt")
        print(f"  → 最佳模型已保存 (val_loss: {val_loss:.6f})")

# =========================================
# 4. 导出 safetensors 与 JSON
# =========================================
print("\n加载最佳模型...")
model.load_state_dict(torch.load(f"{data_dir}/best_model.pt"))
model.eval()
state_dict = model.state_dict()
# 1. 导出供以后纯 Python 分析用的 safetensors
state_dict["y_scale"] = torch.from_numpy(y_max.flatten().astype(np.float32))
save_file(state_dict, f"{data_dir}/jade_sss_weights.safetensors")
print(f"导出完成: {data_dir}/jade_sss_weights.safetensors")
# 2. 内存热转换：直接导出一份供 C++ RTXNS 引擎消费的 JSON
import json

out_data = {"layers": []}
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
json_path = f"{data_dir}/jade_sss_weights.json"
with open(json_path, "w") as f:
    json.dump(out_data, f)
print(f"跨语言同步导出完成: {json_path} (供 C++ RTXNS 加载)")
# =========================================
# 5. 快速质量检查
# =========================================
print("\n抽样验证:")
model.eval()
with torch.no_grad():
    sample_x = torch.from_numpy(X_val[:5]).cuda()
    sample_pred = model(sample_x).cpu().numpy() * y_max
    sample_gt = Y[perm[split : split + 5]]

    # 防止你在第三步忘了改 L1Loss，我们在这里加上 MSE 和 L1 的局部双对比提示
    for i in range(5):
        p = sample_pred[i]
        g = sample_gt[i]
        print(
            f"  样本{i}: 预测=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
            f"真实=({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f})"
        )
