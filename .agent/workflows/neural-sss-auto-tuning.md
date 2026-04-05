---
description: Automated Neural SSS Rendering and Training Tuning Loop
---

# Neural SSS Auto-Tuning Workflow

This workflow automates the pipeline of rendering datasets from Blender, training the Neural SSS PyTorch model, and evaluating its rendering performance.

// turbo-all
1. Render the initial set of EXR data from Blender using the Dragon scene
```bash
blender -b d:/repo/RTXNS/assets/blender/dragon.blend -P d:/repo/RTXNS/python/blender_sss_script.py
```

2. Run the `auto_tune.py` automation script to train models with varying parameters and find the optimal network
```bash
python d:/repo/RTXNS/python/auto_tune.py
```
