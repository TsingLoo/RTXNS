---
description: Automated Neural SSS Rendering and Training Tuning Loop
---

# Neural SSS Auto-Tuning Workflow

This workflow automates the pipeline of rendering datasets from Blender, training the Neural SSS PyTorch model, and evaluating its rendering performance. It serves as an autonomous loop for an AI Agent to iteratively improve realistic subsurface scattering.

// turbo-all
1. Render the initial set of EXR data from Blender using the Dragon scene
```bash
blender -b d:/repo/RTXNS/assets/blender/dragon.blend -P d:/repo/RTXNS/python/blender_sss_script.py
```



2. Run auto_tune.py (which wraps train_sss.py and evaluate_render.py) to systematically sweep hyperparameters, train the current model layout, and output the PSNR validation results.

```
python d:/repo/RTXNS/python/auto_tune.py

```

3. AI Agent Assessment Phase: Check the resulting rendering quality (read the final terminal logs or the psnr.txt value). If the result is not good enough (e.g., PSNR < 35dB or visual fidelity is lacking): Pause and reconsider the architecture. Use your coding tools to edit train_sss.py (network layout, number of branches, activation functions, loss algorithms). You must also reconsider data features and use tools to edit the Blender script (blender_sss_script.py) to extract and bake new geometrical properties (like light depth, world normals, etc.) if required.


4. Iterative Refinement Loop: After you finish editing the python neural networks or Blender rendering engines, go back to Step 1 and begin a completely new loop. Do not stop exploring structural changes until the result is physically accurate and artifacts are eliminated.


