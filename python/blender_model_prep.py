"""
blender_model_prep.py — 通用模型准备模块

对任意 Blender 网格生成 Neural SSS-GGX-MLP 训练/推理所需的全部数据:
  1. 厚度 (Raycast)
  2. AO (Cycles Bake)
  3. 平滑着色 + UV
  4. 相机 + 光源
  5. 几何 Map (Camera Space): Position / Normal / Curvature / Thickness / AO
  6. UV 贴图: thickness_tex / ao_tex / curvature_tex
  7. GLB 导出 (含贴图引用)

用法 (在 blender_sss_script.py 里):
    import blender_model_prep as prep
    ctx = prep.prepare_model(obj, output_dir)
    # ctx 包含 obj, mat, bsdf, nodes, links, center, size, ao_img 等
"""

import bpy
import bmesh
import numpy as np
from mathutils import Vector
import os
import math


# ===========================================================================
# 1. 模型准备 (平滑着色 + UV)
# ===========================================================================
def ensure_smooth_shading(obj):
    """强制开启平滑着色，避免烘焙法线/位置带马赛克方块"""
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.update()
    print("✓ 已开启平滑着色 (Smooth Shading)")


def ensure_uv(obj):
    """如果没有 UV，做一次 Smart UV Project"""
    if len(obj.data.uv_layers) == 0:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        print("✓ UV 展开完成")
    else:
        print(f"✓ 已有 UV: {obj.data.uv_layers[0].name}")


# ===========================================================================
# 2. 厚度计算 (Raycast)
# ===========================================================================
def compute_thickness(obj):
    """
    对每个顶点沿反法线方向射线，计算到背面的距离 → 归一化到 [0,1]
    结果写入顶点色 'Thickness'
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_mesh(obj.evaluated_get(depsgraph).data)
    bm.verts.ensure_lookup_table()

    max_thickness = 0.0
    vert_thickness = []

    for v in bm.verts:
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

    # 写入顶点色
    mesh = obj.data
    if "Thickness" not in mesh.color_attributes:
        mesh.color_attributes.new(name="Thickness", type='FLOAT_COLOR', domain='POINT')

    color_layer = mesh.color_attributes["Thickness"]
    for i in range(len(mesh.vertices)):
        t = vert_thickness[i]
        color_layer.data[i].color = (t, t, t, 1.0)

    print(f"✓ 厚度计算完成 (max_raw = {max_thickness:.4f})")
    return vert_thickness


# ===========================================================================
# 3. AO 烘焙
# ===========================================================================
def bake_ao(obj, output_dir, bake_res=1024, samples=64):
    """烘焙 AO 到 UV 空间纹理，返回 Blender Image 对象"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples

    ao_img = bpy.data.images.new("AO_Map", bake_res, bake_res, alpha=False)
    mat = obj.data.materials[0] if len(obj.data.materials) > 0 else None
    if mat is None:
        mat = bpy.data.materials.new(name="PrepMat")
        obj.data.materials.append(mat)
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
    ao_img.filepath_raw = os.path.join(output_dir, 'ao_uv.exr')
    ao_img.file_format = 'OPEN_EXR'
    ao_img.save()
    print("✓ AO 烘焙完成")
    return ao_img


# ===========================================================================
# 4. 相机设置
# ===========================================================================
def setup_camera(obj, render_res=512):
    """正面相机，自动 framing"""
    scene = bpy.context.scene
    bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(bbox_corners, Vector()) / 8
    size = max((max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)) for i in range(3))

    cam = scene.camera
    if not cam:
        cam_data = bpy.data.cameras.new('Camera')
        cam = bpy.data.objects.new('Camera', cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

    cam.location = center + Vector((0.0, -(size * 2.0), 0.0))
    cam.rotation_euler = (math.radians(90), 0, 0)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = size * 1.2

    scene.render.resolution_x = render_res
    scene.render.resolution_y = render_res
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    np.save(os.path.join(scene.get('output_dir', '/tmp/'), 'camera_pos.npy'),
            np.array(cam.location, dtype=np.float32))

    print(f"✓ 相机设置完成 (center={center}, size={size:.3f})")
    return cam, center, size


# ===========================================================================
# 5. SSS+GGX 材质配置
# ===========================================================================
def setup_sss_material(obj, center, size, base_color=(0.28, 0.58, 0.22, 1.0)):
    """
    创建 Principled BSDF 材质，带完整 SSS + GGX 高光。
    返回 (mat, bsdf, nodes, links)
    """
    mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['IOR'].default_value = 1.62

    # SSS 配置 (翡翠级散射)
    bsdf.inputs['Subsurface Weight'].default_value = 1.0
    bsdf.inputs['Subsurface Scale'].default_value = size * 0.15
    bsdf.inputs['Subsurface Radius'].default_value = (0.8, 1.2, 0.5)

    # SSS-GGX-MLP: 保留完整 GGX 高光
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.5
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.5

    if 'Roughness' in bsdf.inputs:
        bsdf.inputs['Roughness'].default_value = 0.4
    if 'Metallic' in bsdf.inputs:
        bsdf.inputs['Metallic'].default_value = 0.0

    print("✓ SSS+GGX 材质创建完成")
    return mat, bsdf, nodes, links


# ===========================================================================
# 6. Sun Light 配置
# ===========================================================================
def setup_sun_light(output_dir, energy=10.0):
    """配置 Sun Light，返回 light 对象"""
    scene = bpy.context.scene
    light = None
    for o in scene.objects:
        if o.type == 'LIGHT':
            light = o
            break
    if not light:
        light_data = bpy.data.lights.new('Light', type='SUN')
        light = bpy.data.objects.new('Light', light_data)
        scene.collection.objects.link(light)

    light.data.type = 'SUN'
    light.data.energy = energy
    light.data.use_shadow = False

    np.save(os.path.join(output_dir, 'sun_energy.npy'),
            np.array([energy], dtype=np.float32))

    print(f"✓ Sun Light 配置完成 (energy={energy})")
    return light


# ===========================================================================
# 7. Fibonacci 球面采样
# ===========================================================================
def fibonacci_sphere(n):
    """在单位球面上均匀采样 n 个方向"""
    points = []
    golden = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden
        points.append(Vector((
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        )))
    return points


# ===========================================================================
# 8. 几何 Map 渲染 (Camera Space)
# ===========================================================================
def render_geometry_maps(obj, mat, output_dir, center, size, ao_img):
    """
    渲染 Camera Space 几何 Map:
    position_map, normal_map, curvature_map, thickness_map, ao_map
    """
    scene = bpy.context.scene
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    print("渲染几何 Map...")
    nodes.clear()
    geom_node = nodes.new('ShaderNodeNewGeometry')
    emit_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    emit_node.inputs['Strength'].default_value = 1.0

    # --- Position Map ---
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
    print("  ✓ Position Map")

    # --- Normal Map ---
    mul_node = nodes.new('ShaderNodeVectorMath')
    mul_node.operation = 'MULTIPLY'
    mul_node.inputs[1].default_value = (0.5, 0.5, 0.5)
    add_norm = nodes.new('ShaderNodeVectorMath')
    add_norm.operation = 'ADD'
    add_norm.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(geom_node.outputs['Normal'], mul_node.inputs[0])
    links.new(mul_node.outputs[0], add_norm.inputs[0])
    links.new(add_norm.outputs[0], emit_node.inputs['Color'])

    scene.render.filepath = os.path.join(output_dir, 'normal_map.exr')
    bpy.ops.render.render(write_still=True)
    print("  ✓ Normal Map")

    # --- Curvature Map ---
    if 'Pointiness' in geom_node.outputs:
        links.new(geom_node.outputs['Pointiness'], emit_node.inputs['Color'])
        scene.render.filepath = os.path.join(output_dir, 'curvature_map.exr')
        bpy.ops.render.render(write_still=True)
        print("  ✓ Curvature Map")

    # --- Thickness Map (from vertex color) ---
    vc_node = nodes.new('ShaderNodeVertexColor')
    vc_node.layer_name = "Thickness"
    links.new(vc_node.outputs['Color'], emit_node.inputs['Color'])
    scene.render.filepath = os.path.join(output_dir, 'thickness_map.exr')
    bpy.ops.render.render(write_still=True)
    print("  ✓ Thickness Map")

    # --- AO Map (from UV bake) ---
    ao_tex_node = nodes.new('ShaderNodeTexImage')
    ao_tex_node.image = ao_img
    links.new(ao_tex_node.outputs['Color'], emit_node.inputs['Color'])
    scene.render.filepath = os.path.join(output_dir, 'ao_map.exr')
    bpy.ops.render.render(write_still=True)
    print("  ✓ AO Map")


# ===========================================================================
# 9. UV 空间贴图烘焙
# ===========================================================================
def bake_uv_textures(obj, mat, output_dir, ao_img, bake_res=1024):
    """烘焙 UV 空间贴图: thickness_tex, ao_tex, curvature_tex"""
    scene = bpy.context.scene
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64

    images = {}

    # --- Thickness ---
    print("  烘焙 Thickness 纹理...")
    thick_img = bpy.data.images.new("ThicknessBake", bake_res, bake_res, alpha=False)
    thick_img.colorspace_settings.name = 'Non-Color'

    nodes.clear()
    vc_node = nodes.new('ShaderNodeVertexColor')
    vc_node.layer_name = "Thickness"
    emit_node = nodes.new('ShaderNodeEmission')
    out_node = nodes.new('ShaderNodeOutputMaterial')
    tex_target = nodes.new('ShaderNodeTexImage')
    tex_target.name = "BakeTarget"
    tex_target.image = thick_img
    tex_target.select = True
    nodes.active = tex_target
    links.new(vc_node.outputs['Color'], emit_node.inputs['Color'])
    emit_node.inputs['Strength'].default_value = 1.0
    links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])

    bpy.context.scene.cycles.bake_type = 'EMIT'
    bpy.ops.object.bake(type='EMIT')
    thick_img.filepath_raw = os.path.join(output_dir, 'thickness_tex.png')
    thick_img.file_format = 'PNG'
    thick_img.save()
    images['thickness'] = thick_img

    # --- AO ---
    print("  烘焙 AO 纹理...")
    ao_bake_img = bpy.data.images.new("AO_Bake", bake_res, bake_res, alpha=False)
    ao_bake_img.colorspace_settings.name = 'Non-Color'
    tex_target.image = ao_bake_img
    tex_target.select = True
    nodes.active = tex_target

    bpy.context.scene.cycles.bake_type = 'AO'
    bpy.ops.object.bake(type='AO')
    ao_bake_img.filepath_raw = os.path.join(output_dir, 'ao_tex.png')
    ao_bake_img.file_format = 'PNG'
    ao_bake_img.save()
    images['ao'] = ao_bake_img

    # --- Curvature ---
    print("  烘焙 Curvature 纹理...")
    curv_img = bpy.data.images.new("CurvatureBake", bake_res, bake_res, alpha=False)
    curv_img.colorspace_settings.name = 'Non-Color'

    nodes.clear()
    geom_node = nodes.new('ShaderNodeNewGeometry')
    emit_node2 = nodes.new('ShaderNodeEmission')
    out_node2 = nodes.new('ShaderNodeOutputMaterial')
    tex_target2 = nodes.new('ShaderNodeTexImage')
    tex_target2.name = "BakeTarget"
    tex_target2.image = curv_img
    tex_target2.select = True
    nodes.active = tex_target2

    if 'Pointiness' in geom_node.outputs:
        links.new(geom_node.outputs['Pointiness'], emit_node2.inputs['Color'])
    else:
        emit_node2.inputs['Color'].default_value = (0, 0, 0, 1)
    emit_node2.inputs['Strength'].default_value = 1.0
    links.new(emit_node2.outputs['Emission'], out_node2.inputs['Surface'])

    bpy.context.scene.cycles.bake_type = 'EMIT'
    bpy.ops.object.bake(type='EMIT')
    curv_img.filepath_raw = os.path.join(output_dir, 'curvature_tex.png')
    curv_img.file_format = 'PNG'
    curv_img.save()
    images['curvature'] = curv_img

    print("  ✓ UV 贴图烘焙完成")
    return images


# ===========================================================================
# 10. GLB 导出
# ===========================================================================
def export_glb(obj, mat, output_dir, images, size, glb_name='model_sss.glb',
               base_color=(0.28, 0.58, 0.22, 1.0)):
    """组装材质并导出 GLB (含 thickness/AO/curvature 贴图)"""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    out_node = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['IOR'].default_value = 1.62
    bsdf.inputs['Subsurface Weight'].default_value = 1.0
    bsdf.inputs['Subsurface Scale'].default_value = size * 0.15
    bsdf.inputs['Subsurface Radius'].default_value = (0.8, 1.2, 0.5)

    # Thickness → Transmission Weight (KHR_materials_volume)
    if 'thickness' in images:
        thick_node = nodes.new('ShaderNodeTexImage')
        thick_node.image = images['thickness']
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = 0.5
            links.new(thick_node.outputs['Color'], bsdf.inputs['Transmission Weight'])

    # AO → glTF Material Output
    if 'ao' in images:
        gltf_group = None
        for ng in bpy.data.node_groups:
            if ng.name in ('glTF Material Output', 'glTF Settings'):
                gltf_group = ng
                break
        if not gltf_group:
            gltf_group = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
            gltf_group.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')

        gltf_node = nodes.new('ShaderNodeGroup')
        gltf_node.node_tree = gltf_group
        ao_node = nodes.new('ShaderNodeTexImage')
        ao_node.image = images['ao']
        if 'Occlusion' in gltf_node.inputs:
            sep = nodes.new('ShaderNodeSeparateColor')
            links.new(ao_node.outputs['Color'], sep.inputs['Color'])
            links.new(sep.outputs['Red'], gltf_node.inputs['Occlusion'])

    # Curvature → Coat Weight
    if 'curvature' in images:
        curv_node = nodes.new('ShaderNodeTexImage')
        curv_node.image = images['curvature']
        if 'Coat Weight' in bsdf.inputs:
            links.new(curv_node.outputs['Color'], bsdf.inputs['Coat Weight'])
        elif 'Clearcoat' in bsdf.inputs:
            links.new(curv_node.outputs['Color'], bsdf.inputs['Clearcoat'])

    glb_path = os.path.join(output_dir, glb_name)
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format='GLB',
        export_image_format='AUTO',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        use_selection=True,
        export_apply=True,
    )
    print(f"✓ GLB 导出完成: {glb_path}")


# ===========================================================================
# 顶层入口: 一键准备模型
# ===========================================================================
def prepare_model(obj, output_dir, render_res=512, bake_res=1024,
                  base_color=(0.28, 0.58, 0.22, 1.0), sun_energy=10.0):
    """
    一键完成模型准备全流程。返回 context dict 供渲染脚本使用。
    
    参数:
        obj:         Blender 网格对象
        output_dir:  输出目录
        render_res:  渲染分辨率
        bake_res:    UV 烘焙分辨率
        base_color:  基础颜色 (RGBA)
        sun_energy:  Sun Light 能量值
    
    返回:
        dict: { obj, mat, bsdf, nodes, links, center, size, ao_img, light, ... }
    """
    os.makedirs(output_dir, exist_ok=True)
    bpy.context.scene['output_dir'] = output_dir  # 存给 setup_camera 用

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    print("========== 模型准备开始 ==========")

    # Step 1: 平滑着色 + UV
    ensure_smooth_shading(obj)
    ensure_uv(obj)

    # Step 2: 厚度计算
    compute_thickness(obj)

    # Step 3: AO 烘焙
    ao_img = bake_ao(obj, output_dir, bake_res)

    # Step 4: 相机设置
    cam, center, size = setup_camera(obj, render_res)
    # 补存 camera_pos (setup_camera 里可能没存对路径)
    np.save(os.path.join(output_dir, 'camera_pos.npy'),
            np.array(cam.location, dtype=np.float32))

    # Step 5: SSS+GGX 材质
    mat, bsdf, nodes, links = setup_sss_material(obj, center, size, base_color)

    # Step 6: Cycles 渲染配置
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 512
    scene.cycles.use_denoising = True

    # Step 7: Sun Light
    light = setup_sun_light(output_dir, sun_energy)

    print("========== 模型准备完成 ==========\n")

    return {
        'obj': obj,
        'mat': mat,
        'bsdf': bsdf,
        'nodes': nodes,
        'links': links,
        'center': center,
        'size': size,
        'ao_img': ao_img,
        'light': light,
        'cam': cam,
        'output_dir': output_dir,
        'base_color': base_color,
        'render_res': render_res,
        'bake_res': bake_res,
    }


def finalize_model(ctx):
    """准备完成后：渲染几何 Map + UV 贴图烘焙 + GLB 导出"""
    obj = ctx['obj']
    mat = ctx['mat']
    output_dir = ctx['output_dir']
    center = ctx['center']
    size = ctx['size']
    ao_img = ctx['ao_img']
    base_color = ctx['base_color']

    # 几何 Map
    render_geometry_maps(obj, mat, output_dir, center, size, ao_img)

    # UV 贴图
    images = bake_uv_textures(obj, mat, output_dir, ao_img, ctx['bake_res'])

    # GLB 导出
    export_glb(obj, mat, output_dir, images, size, base_color=base_color)

    print("========== 模型最终处理完成 ==========")
    return images

if __name__ == "__main__":
    # 支持在命令行直接运行: blender -b file.blend -P blender_model_prep.py
    import sys
    try:
        # 优先使用活动的物体，如果没有，则查找第一个网格物体
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            for o in bpy.context.scene.objects:
                if o.type == 'MESH':
                    obj = o
                    break
                    
        if not obj:
            raise RuntimeError("在场景中没有找到网格(MESH)物体！")
            
        # 默认将其输出到与 blend 文件同级的一个新文件夹中
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "/tmp"
        out_dir = os.path.join(blend_dir, f"{obj.name}_sss_prepared")
        
        print(f"发现网格物体: {obj.name}，准备导出到 {out_dir}")
        
        ctx = prepare_model(obj, out_dir)
        finalize_model(ctx)
        
        print(f"\n[成功] 脚本独立运行完成，数据存放在: {out_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
