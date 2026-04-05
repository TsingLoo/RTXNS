"""
快速验证：对比 PyTorch 模型 vs JSON 导出的 C++ 推理路径
如果两者结果不一致，说明 export_model_json 有 bug
"""
import json
import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 11

class NeuralSSS(nn.Module):
    def __init__(self):
        super().__init__()
        width = 256
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

def softplus(x):
    return np.log1p(np.exp(x))

def simulate_cpp_inference(json_path, inputs):
    """模拟修正后的 C++ shader 推理：显式残差 skip"""
    with open(json_path) as f:
        data = json.load(f)
    
    layers = data["layers"]
    h = inputs.copy()
    
    for i, layer in enumerate(layers):
        W = np.array(layer["weights"]).reshape(layer["num_outputs"], layer["num_inputs"])
        b = np.array(layer["biases"])
        
        if 1 <= i <= 3:  # 残差层：save → linear → softplus → add saved
            saved = h.copy()
            h = softplus(W @ h + b) + saved
        elif i < len(layers) - 1:  # 普通层：linear → softplus
            h = softplus(W @ h + b)
        else:  # 输出层：linear only
            h = W @ h + b
    
    return np.maximum(h, 0.0)

# 加载模型
data_dir = "c:/tmp/jade_sss_data/"
model = NeuralSSS()
model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location="cpu"))
model.eval()

# 测试几组输入
test_inputs = [
    # NdotL, NdotV, VdotL, thick, ao, curv, wrap, trans, fwd, thin_bl, fresnel
    [0.5, 0.8, 0.3, 0.3, 1.0, 0.0, 0.75, 0.0, 0.008, 0.0, 0.00032],    # 正面照亮
    [-0.5, 0.3, -0.2, 0.1, 1.0, 0.0, 0.25, 0.111, 0.0, 0.45, 0.168],    # 背光 + 薄壁
    [0.0, 0.9, 0.5, 0.5, 0.8, 0.0, 0.5, 0.0, 0.0625, 0.0, 0.00001],     # 侧面
    [-0.8, 0.1, 0.7, 0.05, 1.0, 0.0, 0.1, 0.704, 0.2401, 0.76, 0.590],  # 强背光
]

print("=" * 80)
print("PyTorch 模型 vs C++ JSON 推理对比")
print("=" * 80)

max_diff = 0.0
for i, inp in enumerate(test_inputs):
    # PyTorch
    with torch.no_grad():
        pt_out = model(torch.tensor([inp], dtype=torch.float32))
        pt_out = torch.clamp(pt_out, min=0.0).numpy()[0]
    
    # 模拟 C++
    cpp_out = simulate_cpp_inference(f"{data_dir}/jade_sss_weights.json", np.array(inp))
    
    diff = np.abs(pt_out - cpp_out)
    max_diff = max(max_diff, diff.max())
    
    print(f"\n样本 {i}: {['正面照亮', '背光薄壁', '侧面', '强背光'][i]}")
    print(f"  PyTorch: [{pt_out[0]:.6f}, {pt_out[1]:.6f}, {pt_out[2]:.6f}]")
    print(f"  C++模拟:  [{cpp_out[0]:.6f}, {cpp_out[1]:.6f}, {cpp_out[2]:.6f}]")
    print(f"  差异:     [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}]")

print(f"\n{'=' * 80}")
if max_diff < 0.01:
    print(f"✅ 最大差异 = {max_diff:.8f} — JSON 导出正确!")
else:
    print(f"❌ 最大差异 = {max_diff:.4f} — JSON 导出有 BUG！")
    print("   原因: 残差跳连 fusion (W+I) 在非线性激活下数学不等价！")
    print("   Python: h = softplus(W*h + b) + h  ← skip 在激活函数外面")
    print("   C++:    h = softplus((W+I)*h + b)  ← skip 被融进了激活函数里面")
