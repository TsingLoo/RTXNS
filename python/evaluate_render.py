import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import cv2

# =========================================
# 0. 数据路径与辅助函数 (按需修改路径)
# =========================================
data_dir = "c:/tmp/jade_sss_data/"

def load_exr_channels(filepath, channels=('R', 'G', 'B', 'A')):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load {filepath}")
    
    if img.shape[2] == 4:
        return img[:, :, [2, 1, 0, 3]] # BGRA -> RGBA
    else:
        return img[:, :, [2, 1, 0]]    # BGR -> RGB

# =========================================
# 1. 网络结构
# =========================================
class FrequencyEncoder(nn.Module):
    def __init__(self, input_dim=7, freq_bands=4):
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
        width = 128
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
# 2. 渲染主函数
# =========================================
def render_offline_preview(test_idx=0):
    print(f"\n[验证] 开始使用确定的 safetensors 权重渲染视角 {test_idx:04d} ...")
    
    # 1. 初始化并加载权重
    model = NeuralSSS().cuda()
    model.eval()
    
    safetensors_path = os.path.join(data_dir, "jade_sss_weights.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Error: 找不到权重文件 {safetensors_path}")
        return
        
    state_dict = load_file(safetensors_path)
    # 把放进去的 y_scale 单独取出来
    y_max = state_dict.pop("y_scale").cpu().numpy()
    model.load_state_dict(state_dict)
    
    # 2. 加载 G-Buffer
    print("加载 G-Buffer 图像与几何数据...")
    try:
        render_rgba = load_exr_channels(os.path.join(data_dir, f'render_{test_idx:04d}.exr'))
        light_dir = np.load(os.path.join(data_dir, f'lightdir_{test_idx:04d}.npy'))
        
        # 现在统统改用你手头上的 .exr 文件！
        def safe_load_exr(file_base):
            # 优先找带序列号的，找不到就找没序列号的通用图
            path = os.path.join(data_dir, f'{file_base}_{test_idx:04d}.exr')
            if not os.path.exists(path):
                path = os.path.join(data_dir, f'{file_base}.exr')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {file_base}.exr")
            return load_exr_channels(path)
            
        position_map = safe_load_exr('position_map')[:, :, :3]
        normal_map = safe_load_exr('normal_map')[:, :, :3]
        
        # 厚度和 AO 可能导出成通道，稳妥起见统一取第一通道，如果报错再说
        _thick = safe_load_exr('thickness_map')
        thickness_map = _thick[:, :, 0] if _thick.ndim == 3 else _thick
        
        _ao = safe_load_exr('ao_map')
        ao_map = _ao[:, :, 0] if _ao.ndim == 3 else _ao
        
        cam_pos_file = os.path.join(data_dir, 'camera_pos.npy')
        if os.path.exists(cam_pos_file):
            cam_pos = np.load(cam_pos_file)
        else:
            print(f"\n⚠️ 警告: 未找到 {cam_pos_file}！")
            cam_pos = np.array([0.0, -5.0, 1.0]) 

    except Exception as e:
        print(f"\n❌ 数据加载失败！请确保以下变量文件确实保存在 data_dir 中:\n{e}\n\n-> 如果你当时没有把逐像素的 position_map.npy 单独存成文件，你必须改用你原来的脚本去套用此逻辑！")
        return

    L = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    h, w = render_rgba.shape[:2]
    
    img_pred = np.zeros((h, w, 3), dtype=np.float32)
    img_gt = np.zeros((h, w, 3), dtype=np.float32)
    
    batch_features = []
    batch_pixels = []
    
    # 3. 逐个像素提取特征
    for y in range(h):
        for x in range(w):
            a = render_rgba[y, x, 3]
            img_gt[y, x] = render_rgba[y, x, :3] 
            
            if a < 0.5:
                continue
                
            pixel_pos = position_map[y, x]
            V = cam_pos - pixel_pos
            V_len = np.linalg.norm(V)
            if V_len < 1e-8: continue
            V = V / V_len
            
            normal = normal_map[y, x]
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8: continue
            normal = normal / norm_len
            
            NdotL = np.dot(normal, L)
            NdotV = max(np.dot(normal, V), 0.0)
            VdotL = np.dot(V, L)
            thick = thickness_map[y, x]
            ao = ao_map[y, x]
            
            # 严格保持与 C++ 一致的特征顺序绑定常量
            feat = [NdotL, NdotV, VdotL, thick, ao, 0.15, 0.0]
            batch_features.append(feat)
            batch_pixels.append((y, x))
            
    print(f"共提取 {len(batch_features)} 个非透明前景像素, 开始分块推理...")
    X_tensor = torch.tensor(batch_features, dtype=torch.float32).cuda()
    
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_tensor), 65536):
            chunk = X_tensor[i:i+65536]
            preds.append(model(chunk))
        preds = torch.cat(preds, dim=0).cpu().numpy()
    
    # 4. 根据 y_max 还原色彩
    preds = preds * y_max
    
    for i, (y, x) in enumerate(batch_pixels):
        img_pred[y, x] = preds[i]

    # ACES 色调映射
    def simple_aces(x):
        x = np.maximum(0, x - 0.004)
        return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

    pred_show = simple_aces(img_pred)
    gt_show = simple_aces(img_gt)

    # Calculate PSNR
    mse = np.mean((pred_show - gt_show) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    print(f"\n=============================================")
    print(f"Final Rendering Quality (PSNR): {psnr:.2f} dB")
    print(f"=============================================\n")
    
    with open(os.path.join(data_dir, "psnr.txt"), "w") as f:
        f.write(str(psnr))

    # 5. 绘制并保存
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(pred_show, 0, 1))
    plt.title("PyTorch MLP Render", fontsize=16)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(gt_show, 0, 1))
    plt.title("Blender Ground Truth", fontsize=16)
    plt.axis("off")
    
    plt.tight_layout()
    output_png = os.path.join(data_dir, "render_comparison.png")
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    print(f"任务完成！请打开 {output_png} 查看渲染结果！")

if __name__ == "__main__":
    render_offline_preview(0)
