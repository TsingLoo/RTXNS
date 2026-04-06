import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =========================================
# 0. 数据路径与辅助函数 (按需修改路径)
# =========================================
data_dir = "c:/tmp/jade_sss_data/"
INPUT_DIM = 15

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
# 1. 网络结构 (残差 + SSS 特征，必须与 train_sss.py 一致)
# =========================================
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
        h = self.block1(h) + h
        h = self.block2(h) + h
        h = self.block3(h) + h
        return self.output(h)

# =========================================
# 2. 连续渲染循环
# =========================================
def render_live_preview(test_idx=0):
    print(f"\n[实时验证] 监听 best_model.pt 的更新并自动刷新渲染...")
    
    # 初始化网络
    model = NeuralSSS().cuda()
    model.eval()
    
    # 2. 加载 G-Buffer
    print("加载 G-Buffer 图像与几何数据...")
    try:
        render_rgba = load_exr_channels(os.path.join(data_dir, f'render_{test_idx:04d}.exr'))
        light_dir = np.load(os.path.join(data_dir, f'lightdir_{test_idx:04d}.npy'))
        
        # 加载材质参数
        mat_path = os.path.join(data_dir, f'matparams_{test_idx:04d}.npy')
        if os.path.exists(mat_path):
            mat_params = np.load(mat_path)
            render_roughness, render_metallic, render_specular = mat_params
        else:
            render_roughness, render_metallic, render_specular = 0.4, 0.0, 0.5
        
        def safe_load_exr(file_base):
            path = os.path.join(data_dir, f'{file_base}_{test_idx:04d}.exr')
            if not os.path.exists(path):
                path = os.path.join(data_dir, f'{file_base}.exr')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {file_base}.exr")
            return load_exr_channels(path)
            
        position_map = safe_load_exr('position_map')[:, :, :3]
        normal_map = safe_load_exr('normal_map')[:, :, :3]
        
        _thick = safe_load_exr('thickness_map')
        thickness_map = _thick[:, :, 0] if _thick.ndim == 3 else _thick
        
        _ao = safe_load_exr('ao_map')
        ao_map = _ao[:, :, 0] if _ao.ndim == 3 else _ao
        
        try:
            _curv = safe_load_exr('curvature_map')
            curvature_map = _curv[:, :, 0] if _curv.ndim == 3 else _curv
        except FileNotFoundError:
            curvature_map = np.zeros_like(ao_map)
        
        cam_pos_file = os.path.join(data_dir, 'camera_pos.npy')
        if os.path.exists(cam_pos_file):
            cam_pos = np.load(cam_pos_file)
        else:
            cam_pos = np.array([0.0, -5.0, 1.0]) 
            
        center = np.load(os.path.join(data_dir, 'bbox_center.npy'))
        size = np.load(os.path.join(data_dir, 'bbox_size.npy'))[0]
        pos_map = (position_map - 0.5) * size + center
        norm_map = (normal_map - 0.5) * 2.0

    except Exception as e:
        print(f"\n❌ 数据加载失败！请确保预计算映射在 data_dir 中:\n{e}")
        return

    L = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    h, w = render_rgba.shape[:2]
    
    img_pred = np.zeros((h, w, 3), dtype=np.float32)
    img_gt = np.zeros((h, w, 3), dtype=np.float32)
    
    batch_features = []
    batch_pixels = []
    
    for y in range(h):
        for x in range(w):
            a = render_rgba[y, x, 3]
            img_gt[y, x] = render_rgba[y, x, :3] 
            
            if a < 0.99:
                continue
                
            pixel_pos = pos_map[y, x]
            V = cam_pos - pixel_pos
            V_len = np.linalg.norm(V)
            if V_len < 1e-8: continue
            V = V / V_len
            
            normal = norm_map[y, x]
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8: continue
            normal = normal / norm_len
            
            NdotL = np.dot(normal, L)
            NdotV = max(np.dot(normal, V), 0.0)
            VdotL = np.dot(V, L)
            
            # SSS-GGX-MLP: Half Vector for specular
            H = V + L
            H_len = np.linalg.norm(H)
            if H_len < 1e-8:
                NdotH = 0.0
            else:
                H = H / H_len
                NdotH = max(np.dot(normal, H), 0.0)
            
            thick = thickness_map[y, x]
            ao = ao_map[y, x]
            curv = curvature_map[y, x]
            
            # ======== SSS 专属特征 (与 train_sss.py 一致) ========
            wrap_lighting = max(0.0, min(1.0, NdotL * 0.5 + 0.5))
            transmission = np.exp(-thick * 3.0) * max(-NdotL, 0.0)
            forward_scatter = max(VdotL, 0.0) ** 4.0
            thin_backlight = (1.0 - thick) * max(-NdotL, 0.0)
            fresnel = (1.0 - NdotV) ** 5.0
            
            feat = [NdotL, NdotV, NdotH, VdotL,
                    render_roughness, render_metallic, render_specular,
                    thick, ao, curv,
                    wrap_lighting, transmission, forward_scatter, thin_backlight, fresnel]
            batch_features.append(feat)
            batch_pixels.append((y, x))
            
    print(f"共提取 {len(batch_features)} 个像素, 准备进入轮询循环...")
    X_tensor = torch.tensor(batch_features, dtype=torch.float32).cuda()
    
    base_color = np.array([0.28, 0.58, 0.22], dtype=np.float32)

    # 加载 sun_energy: MLP 输出是单位光强 transmittance，需要乘回 energy 才能和 GT 对比
    sun_energy_file = os.path.join(data_dir, 'sun_energy.npy')
    if os.path.exists(sun_energy_file):
        sun_energy = np.load(sun_energy_file)[0]
        print(f"已加载 Sun energy = {sun_energy}，MLP 输出将乘以此值以匹配 GT")
    else:
        sun_energy = 1.0

    def simple_aces(x):
        x = np.maximum(0, x - 0.004)
        return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Live Training Preview (Remote)')
    im_pred = axes[0].imshow(np.zeros((h, w, 3)))
    axes[0].set_title("MLP Live Output")
    axes[0].axis("off")
    gt_show = simple_aces(img_gt)
    axes[1].imshow(np.clip(gt_show, 0, 1))
    axes[1].set_title("Blender Ground Truth")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show(block=False)

    best_model_path = os.path.join(data_dir, "best_model.pt")
    last_mtime = 0

    while True:
        try:
            if os.path.exists(best_model_path):
                current_mtime = os.path.getmtime(best_model_path)
                if current_mtime > last_mtime:
                    last_mtime = current_mtime
                    try:
                        time.sleep(0.1) # 稍等文件写入完毕
                        model.load_state_dict(torch.load(best_model_path))
                        
                        with torch.no_grad():
                            preds = []
                            for i in range(0, len(X_tensor), 65536):
                                chunk = X_tensor[i:i+65536]
                                preds.append(model(chunk))
                            preds_hdr = torch.cat(preds, dim=0).cpu().numpy()
                        
                        preds_hdr = np.maximum(preds_hdr, 0.0)
                        # MLP 输出是单位光强 transmittance，乘回 sun_energy + base_color 以匹配 GT
                        preds_rgb = preds_hdr * base_color * sun_energy
                        
                        for i, (py, px) in enumerate(batch_pixels):
                            img_pred[py, px] = preds_rgb[i]

                        pred_show = simple_aces(img_pred)
                        im_pred.set_data(np.clip(pred_show, 0, 1))
                        
                        mse = np.mean((pred_show - gt_show) ** 2)
                        psnr = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
                        axes[0].set_title(f"MLP Live Output\nPSNR: {psnr:.2f} dB")
                        print(f"[{time.strftime('%H:%M:%S')}] 侦测到权重更新 => 刷新画面 (PSNR: {psnr:.2f} dB)")
                    except Exception as e:
                        print(f"读取或推理期间出错 (可能文件正在被占用): {e}")

            plt.pause(1.0)
            
            if not plt.fignum_exists(fig.number):
                break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    render_live_preview(0)
