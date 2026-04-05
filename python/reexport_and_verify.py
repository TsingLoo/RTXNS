import json, numpy as np, torch, torch.nn as nn

INPUT_DIM = 11
class NeuralSSS(nn.Module):
    def __init__(self):
        super().__init__()
        w = 256
        self.input_proj = nn.Linear(INPUT_DIM, w)
        self.act = nn.Softplus()
        self.block1 = nn.Sequential(nn.Linear(w, w), nn.Softplus())
        self.block2 = nn.Sequential(nn.Linear(w, w), nn.Softplus())
        self.block3 = nn.Sequential(nn.Linear(w, w), nn.Softplus())
        self.output = nn.Linear(w, 3)
    def forward(self, x):
        h = self.act(self.input_proj(x))
        h = self.block1(h) + h; h = self.block2(h) + h; h = self.block3(h) + h
        return self.output(h)

data_dir = "c:/tmp/jade_sss_data"
model = NeuralSSS()
model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location="cpu"))
model.eval()
sd = model.state_dict()

out_data = {"layers": [], "input_dim": INPUT_DIM, "freq_bands": 0}
w0, b0 = sd["input_proj.weight"].numpy(), sd["input_proj.bias"].numpy()
out_data["layers"].append({"num_inputs": int(w0.shape[1]), "num_outputs": int(w0.shape[0]),
                           "weights": w0.flatten().tolist(), "biases": b0.flatten().tolist()})
for idx in range(1, 4):
    w = sd[f"block{idx}.0.weight"].numpy()
    b = sd[f"block{idx}.0.bias"].numpy()
    out_data["layers"].append({"num_inputs": int(w.shape[1]), "num_outputs": int(w.shape[0]),
                               "weights": w.flatten().tolist(), "biases": b.flatten().tolist()})
wo, bo = sd["output.weight"].numpy(), sd["output.bias"].numpy()
out_data["layers"].append({"num_inputs": int(wo.shape[1]), "num_outputs": int(wo.shape[0]),
                           "weights": wo.flatten().tolist(), "biases": bo.flatten().tolist()})
out_data["y_scale"] = [1.0, 1.0, 1.0]
out_data["base_color"] = [0.28, 0.58, 0.22]

path = f"{data_dir}/jade_sss_weights.json"
with open(path, "w") as f:
    json.dump(out_data, f)

# Verify
with open(path) as f:
    check = json.load(f)
w1_json = np.array(check["layers"][1]["weights"]).reshape(256, 256)
w1_sd = sd["block1.0.weight"].numpy()
print(f"Layer1 diff: {np.abs(w1_sd - w1_json).max():.8f}")
print(f"Num layers in JSON: {len(check['layers'])}")

# Full inference test
def softplus(x): return np.log1p(np.exp(np.clip(x, -50, 50)))
inp = np.array([0.5, 0.8, 0.3, 0.3, 1.0, 0.0, 0.75, 0.0, 0.008, 0.0, 0.00032], dtype=np.float32)

with torch.no_grad():
    pt = model(torch.tensor([inp]))[0].numpy()
    pt = np.maximum(pt, 0)

h = inp.copy()
for i, L in enumerate(check["layers"]):
    W = np.array(L["weights"]).reshape(L["num_outputs"], L["num_inputs"])
    b = np.array(L["biases"])
    if 1 <= i <= 3:
        saved = h.copy()
        h = softplus(W @ h + b) + saved
    elif i < len(check["layers"]) - 1:
        h = softplus(W @ h + b)
    else:
        h = W @ h + b
cpp = np.maximum(h, 0)

diff = np.abs(pt - cpp).max()
print(f"PyTorch:  {pt}")
print(f"C++ sim:  {cpp}")
print(f"Max diff: {diff:.8f}")
print("PASS" if diff < 0.01 else "FAIL")
