import bpy
import bmesh
import numpy as np
from mathutils import Vector, Euler
import os
import math

output_dir = 'c:/tmp/jade_sss_data/'
os.makedirs(output_dir, exist_ok=True)

# 确保选中了龙模型
obj = bpy.data.objects.get('Dragon')
if not obj:
    obj = bpy.context.active_object
bpy.context.view_layer.objects.active = obj

# ============================================
# 0. 准备工作：自动展 UV 与 强制平滑着色 (破解马赛克现象的关键！)
# ============================================
print("--- 1. 准备模型 ---")
# 强制开启平滑着色，避免烘焙出来的 Normal 和 Position 带马赛克方块！
for p in obj.data.polygons:
    p.use_smooth = True
# 刷新法线信息
obj.data.update()
print("已强行开启平滑着色 (Smooth Shading)")

# 如果没有UV，做一次Smart UV Project
if len(obj.data.uv_layers) == 0:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    print("UV展开完成")
else:
    print(f"已有UV查收: {obj.data.uv_layers[0].name}")


# ============================================
# 1. 烘焙厚度图
# ============================================
print("--- 2.计算并烘焙厚度 ---")
depsgraph = bpy.context.evaluated_depsgraph_get()
bm = bmesh.new()
bm.from_mesh(obj.evaluated_get(depsgraph).data)
bm.verts.ensure_lookup_table()

total = len(bm.verts)
max_thickness = 0.0
vert_thickness = []

for i, v in enumerate(bm.verts):
    origin = v.co + (-v.normal) * 0.001 
    direction = -v.normal
    hit, loc, nor, face_idx = obj.ray_cast(origin, direction)
    dist = (v.co - loc).length if hit else 0.0
    vert_thickness.append(dist)
    max_thickness = max(max_thickness, dist)

bm.free()

if max_thickness > 0:
    vert_thickness = np.array(vert_thickness) / max_thickness
else:
    vert_thickness = np.array(vert_thickness)

mesh = obj.data
if "Thickness" not in mesh.color_attributes:
    mesh.color_attributes.new(name="Thickness", type='FLOAT_COLOR', domain='POINT')

color_layer = mesh.color_attributes["Thickness"]
for i in range(len(mesh.vertices)):
    t = vert_thickness[i]
    color_layer.data[i].color = (t, t, t, 1.0)
print("厚度已计算并写入顶点色 'Thickness'")


# ============================================
# 2. 烘焙 AO
# ============================================
print("--- 3. 烘焙 AO ---")
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64

ao_img = bpy.data.images.new("AO_Map", 1024, 1024, alpha=False)
mat = obj.data.materials[0] if len(obj.data.materials) > 0 else obj.data.materials.append(bpy.data.materials.new(name="JadeMat"))
mat = obj.data.materials[0]
mat.use_nodes = True
nodes = mat.node_tree.nodes

tex_node = nodes.get("BakeTarget") or nodes.new('ShaderNodeTexImage')
tex_node.name = "BakeTarget"
tex_node.image = ao_img
tex_node.select = True
nodes.active = tex_node

bpy.context.scene.cycles.bake_type = 'AO'
bpy.ops.object.bake(type='AO')
ao_img.filepath_raw = os.path.join(output_dir, 'ao_map.exr')
ao_img.file_format = 'OPEN_EXR'
ao_img.save()
print("AO 烘焙并保存完成")


# ============================================
# 3. 配置纯净 SSS 材质与相机
# ============================================
print("--- 4. 配置场景并渲染图像 ---")
render_res = 512
n_light_angles = 60
cycles_samples = 512

links = mat.node_tree.links
nodes.clear()
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
output = nodes.new('ShaderNodeOutputMaterial')
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

bsdf.inputs['Base Color'].default_value = (0.28, 0.58, 0.22, 1.0)
bsdf.inputs['IOR'].default_value = 1.62
bsdf.inputs['Subsurface Weight'].default_value = 1.0
bsdf.inputs['Subsurface Scale'].default_value = 0.8
bsdf.inputs['Subsurface Radius'].default_value = (0.8, 1.2, 0.5)

# 🛑 最核心改动：物理级阉割所有高光，只让它渲染通透的 SSS 漫反射底盘
if 'Specular IOR Level' in bsdf.inputs:
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
elif 'Specular' in bsdf.inputs:
    bsdf.inputs['Specular'].default_value = 0.0

if 'Roughness' in bsdf.inputs:
    bsdf.inputs['Roughness'].default_value = 1.0

scene.cycles.samples = cycles_samples
scene.cycles.use_denoising = True
scene.render.resolution_x = render_res
scene.render.resolution_y = render_res
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.view_settings.view_transform = 'Raw'

cam = bpy.data.objects.get('Camera')
if not cam:
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    scene.collection.objects.link(cam)
scene.camera = cam

bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
center = sum(bbox_corners, Vector()) / 8
size = max((max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)) for i in range(3))

cam_dist = size * 2.5
cam.location = center + Vector((0, -cam_dist, cam_dist * 0.5))
direction = center - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

np.save(os.path.join(output_dir, 'camera_pos.npy'), np.array(cam.location, dtype=np.float32))


# ============================================
# 4. 循环打光渲染 (漫反射图)
# ============================================
light = bpy.data.objects.get('Light')
if not light:
    light_data = bpy.data.lights.new('Light', type='POINT')
    light = bpy.data.objects.new('Light', light_data)
    scene.collection.objects.link(light)

light.data.type = 'POINT'
light.data.energy = 1000 * (size ** 2)
# 彻底关闭阴影投射，确保渲染出的只包含纯粹的SSS散射数据
light.data.use_shadow = False
light_radius = size * 3

def fibonacci_sphere(n):
    points = []
    golden = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2*(i + 0.5)/n)
        phi = 2 * math.pi * i / golden
        points.append(Vector((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))))
    return points

light_dirs = fibonacci_sphere(n_light_angles)

print(f"开始渲染 {n_light_angles} 张去除高光纯净 SSS 的视角...")
for idx, ld in enumerate(light_dirs):
    light.location = center + ld * light_radius
    scene.render.filepath = os.path.join(output_dir, f'render_{idx:04d}.exr')
    bpy.ops.render.render(write_still=True)
    np.save(os.path.join(output_dir, f'lightdir_{idx:04d}.npy'), np.array(ld, dtype=np.float32))


# ============================================
# 5. 输出附带的 Position Map (平滑)
# ============================================
print("--- 5. 渲染基础属性 Map ---")
nodes.clear()
geom_node = nodes.new('ShaderNodeNewGeometry')
emit_node = nodes.new('ShaderNodeEmission')
output_node = nodes.new('ShaderNodeOutputMaterial')
emit_node.inputs['Strength'].default_value = 1.0

# For Position: scaled_pos = (pos - center) / size + 0.5
sub_node = nodes.new('ShaderNodeVectorMath')
sub_node.operation = 'SUBTRACT'
sub_node.inputs[1].default_value = center

div_node = nodes.new('ShaderNodeVectorMath')
div_node.operation = 'DIVIDE'
div_node.inputs[1].default_value = (size, size, size)

add_node = nodes.new('ShaderNodeVectorMath')
add_node.operation = 'ADD'
add_node.inputs[1].default_value = (0.5, 0.5, 0.5)

links.new(geom_node.outputs['Position'], sub_node.inputs[0])
links.new(sub_node.outputs[0], div_node.inputs[0])
links.new(div_node.outputs[0], add_node.inputs[0])
links.new(add_node.outputs[0], emit_node.inputs['Color'])
links.new(emit_node.outputs['Emission'], output_node.inputs['Surface'])

scene.cycles.samples = 1
scene.render.image_settings.color_mode = 'RGB'
scene.render.filepath = os.path.join(output_dir, 'position_map.exr')
bpy.ops.render.render(write_still=True)
np.save(os.path.join(output_dir, 'bbox_center.npy'), np.array(center, dtype=np.float32))
np.save(os.path.join(output_dir, 'bbox_size.npy'), np.array([size], dtype=np.float32))
print("Position Map 烘焙完成！")


# For Normal: scaled_norm = (norm * 0.5) + 0.5
mul_node = nodes.new('ShaderNodeVectorMath')
mul_node.operation = 'MULTIPLY'
mul_node.inputs[1].default_value = (0.5, 0.5, 0.5)

add_norm_node = nodes.new('ShaderNodeVectorMath')
add_norm_node.operation = 'ADD'
add_norm_node.inputs[1].default_value = (0.5, 0.5, 0.5)

links.new(geom_node.outputs['Normal'], mul_node.inputs[0])
links.new(mul_node.outputs[0], add_norm_node.inputs[0])
links.new(add_norm_node.outputs[0], emit_node.inputs['Color'])

scene.render.filepath = os.path.join(output_dir, 'normal_map.exr')
bpy.ops.render.render(write_still=True)
print("Normal Map 烘焙完成！")

# 烘焙曲率 (Curvature / Pointiness) 图
if 'Pointiness' in geom_node.outputs:
    links.new(geom_node.outputs['Pointiness'], emit_node.inputs['Color'])
    scene.render.filepath = os.path.join(output_dir, 'curvature_map.exr')
    bpy.ops.render.render(write_still=True)
    print("Curvature/Pointiness Map 烘焙完成！")

print("========== 全部生成任务圆满完成 ==========")
