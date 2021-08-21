from os import makedirs
from os.path import dirname, exists
import sys
import json
import argparse
import numpy as np
import bpy
from mathutils import Matrix


def set_cycles(w=None, h=None, n_samples=None):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    cycles.use_progressive_refine = True
    if n_samples is not None:
        cycles.samples = n_samples
    cycles.max_bounces = 100
    cycles.min_bounces = 10
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5

    # Ensure no background node
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    scene.render.tile_x = 16
    scene.render.tile_y = 16
    if w is not None:
        scene.render.resolution_x = w
    if h is not None:
        scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'


def add_object(model_path, rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
               trans_vec=(0, 0, 0), scale=1, name=None):
    # Import
    if model_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=model_path, axis_forward='-Z', axis_up='Y')
    else:
        raise NotImplementedError("Importing model of this type")

    obj_list = []
    for i, obj in enumerate(bpy.context.selected_objects):
        # Rename
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # Compute world matrix
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # don't scale here
        obj.matrix_world = trans_4x4 @ rot_4x4 @ scale_4x4

        # Scale
        obj.scale = (scale, scale, scale)

        obj_list.append(obj)

    if len(obj_list) == 1:
        return obj_list[0]
    else:
        return obj_list


def add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None,
               proj_model='PERSP', f=35, sensor_fit='HORIZONTAL',
               sensor_width=32, sensor_height=18):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    return cam


def render_to_file(outpath):
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)

    # Set active camera, just in case
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    # Render
    bpy.context.scene.render.filepath = outpath
    bpy.ops.render.render(write_still=True)


def render(data, output_path):
    w, h = data['img_size']
    set_cycles(w=w, h=h, n_samples=50)

    # Remove all default objects
    for obj in bpy.data.objects:
        obj.select_set(True)
    bpy.ops.object.delete()

    # Object
    obj = add_object(data['model'], data['rot_mat'],
                     data['trans_mat'], name='object')

    # Lighting
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 0.9

    # Camera
    camera = add_camera((0, 0, 0), (0, np.pi, 0), 'camera', 'PERSP',
                        data['focal_length'], 'HORIZONTAL', 32)
    camera.data.clip_end = 1e10

    render_to_file(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_idx', type=int, default=0,
                        help='index of annotion')
    parser.add_argument('--output_path', type=str, default='./demo.png',
                        help='output image path')
    if '--' not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_args(argv)

    data_list = json.load(open('pix3d.json'))
    render(data_list[args.anno_idx], args.output_path)
    print('Original Image:', data_list[args.anno_idx]['img'])
    print('Saved to:', args.output_path)

