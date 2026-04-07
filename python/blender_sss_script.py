"""
blender_sss_script.py — SSS-GGX-MLP 数据集生成器

使用 blender_model_prep.py 准备模型, 然后渲染多光方向×多材质变体的训练数据。

用法:
    blender --background scene.blend --python blender_sss_script.py

可配置参数 (修改下方 CONFIG 区块):
    - obj_name:          Blender 场景里的网格名称
    - output_dir:        输出目录
    - n_light_angles:    Fibonacci 球面采样的光方向数
    - n_material_variants: 每个光方向渲染几种材质配置
    - render_res:        渲染分辨率
    - cycles_samples:    Cycles 采样数
    - base_color:        基础颜色 (RGBA)
    - sun_energy:        Sun Light 能量
"""

import bpy
import numpy as np
import os
import sys
import random

# 确保能导入同目录模块
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import blender_model_prep as prep

# ============================================
# CONFIG — 数据集生成参数
# ============================================
obj_name = 'Dragon'          # Blender 场景中的网格名称 (改成你的模型名)
output_dir = 'c:/tmp/jade_sss_data/'
n_light_angles = 60          # 光源方向数
n_material_variants = 3      # 每个光方向的材质变体数
render_res = 512             # 渲染分辨率
cycles_samples = 512         # Cycles 采样数
base_color = (0.28, 0.58, 0.22, 1.0)  # 翡翠绿
sun_energy = 10.0            # Sun Light 能量 (训练时会自动归一化)
random_seed = 42             # 材质随机种子 (可复现)

# ============================================
# 1. 获取模型对象
# ============================================
obj = bpy.data.objects.get(obj_name)
if not obj:
    obj = bpy.context.active_object
if not obj or obj.type != 'MESH':
    raise RuntimeError(f"找不到网格对象 '{obj_name}'，请检查 Blender 场景")

print(f"使用模型: {obj.name}")

# ============================================
# 2. 准备模型 (厚度/AO/相机/材质/灯光)
# ============================================
ctx = prep.prepare_model(
    obj,
    output_dir,
    render_res=render_res,
    base_color=base_color,
    sun_energy=sun_energy,
)

# 设置 Cycles 采样数
bpy.context.scene.cycles.samples = cycles_samples
bpy.context.scene.cycles.use_denoising = True

# ============================================
# 3. 渲染 SSS+GGX 训练数据
# ============================================
light_dirs = prep.fibonacci_sphere(n_light_angles)
total_renders = n_light_angles * n_material_variants
random.seed(random_seed)

bsdf = ctx['bsdf']
light = ctx['light']
scene = bpy.context.scene

print(f"\n开始渲染 {total_renders} 张 SSS+GGX 训练数据")
print(f"  ({n_light_angles} 光方向 × {n_material_variants} 材质变体)")
print(f"  分辨率={render_res}, Cycles={cycles_samples}, Energy={sun_energy}\n")

render_idx = 0
for light_idx, ld in enumerate(light_dirs):
    # 设置光方向
    light_forward = -ld
    light.rotation_euler = light_forward.to_track_quat('-Z', 'Y').to_euler()

    for mat_var in range(n_material_variants):
        # 随机材质参数
        r_roughness = random.uniform(0.1, 0.9)
        r_metallic = random.uniform(0.0, 0.05)
        r_specular = random.uniform(0.3, 0.7)

        # 应用到 Blender BSDF
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = r_roughness
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = r_metallic
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = r_specular
        elif 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = r_specular

        # 渲染
        scene.render.filepath = os.path.join(output_dir, f'render_{render_idx:04d}.exr')
        bpy.ops.render.render(write_still=True)

        # 保存参数
        np.save(os.path.join(output_dir, f'lightdir_{render_idx:04d}.npy'),
                np.array(ld, dtype=np.float32))
        np.save(os.path.join(output_dir, f'matparams_{render_idx:04d}.npy'),
                np.array([r_roughness, r_metallic, r_specular], dtype=np.float32))

        print(f"  [{render_idx+1}/{total_renders}] L={light_idx}, "
              f"rough={r_roughness:.2f}, metal={r_metallic:.2f}, spec={r_specular:.2f}")
        render_idx += 1

# 恢复默认材质
if 'Roughness' in bsdf.inputs:
    bsdf.inputs['Roughness'].default_value = 0.4
if 'Metallic' in bsdf.inputs:
    bsdf.inputs['Metallic'].default_value = 0.0

# ============================================
# 4. 渲染几何 Map + UV 贴图 + GLB 导出
# ============================================
print("\n--- 渲染几何 Map 和 UV 贴图 ---")
images = prep.finalize_model(ctx)

print(f"\n========== 数据集生成完成 ==========")
print(f"  总渲染数: {total_renders}")
print(f"  输出目录: {output_dir}")
print(f"  文件清单:")
print(f"    - render_XXXX.exr     × {total_renders}")
print(f"    - lightdir_XXXX.npy   × {total_renders}")
print(f"    - matparams_XXXX.npy  × {total_renders}")
print(f"    - position/normal/thickness/ao/curvature_map.exr")
print(f"    - thickness/ao/curvature_tex.png")
print(f"    - sun_energy.npy, camera_pos.npy, bbox_*.npy")
print(f"    - model_sss.glb")
