#!/usr/bin/env python3
import cv2
import os 
import json
import pyrender
import numpy as np
import sys
import trimesh
import matplotlib.pyplot as plt
import timeit
import argparse
import yaml

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


def lookAt(eye, target, up):
    # eye is from
    # target is to
    # expects numpy arrays
    f = eye - target
    f = f/np.linalg.norm(f)

    s = np.cross(up, f)
    s = s/np.linalg.norm(s)
    u = np.cross(f, s)
    u = u/np.linalg.norm(u)

    tx = np.dot(s, eye.T)
    ty = np.dot(u, eye.T)
    tz = np.dot(f, eye.T)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = f
    m[:, 3] = [tx, ty, tz, 1]

    return m


def load_meshes(mesh_dir, ren):
    for mesh_now in os.listdir(mesh_dir):
        mesh_path_now = os.path.join(mesh_dir, mesh_now)
        if mesh_now[-4:] != '.ply':
            continue
        mesh_id = int(mesh_now[4:-4])
        ren.add_object(mesh_id, mesh_path_now)
        mesh_id += 1   
    return ren

def occlusion_mask_from_segmap(segmap, obj_id):
    mask = np.zeros_like(segmap)
    mask[segmap == obj_id] = 255
    return mask

def render_mask(obj_info, cam_intr, r, meshes):
    obj_id = int(obj_info["obj_id"])

    R = np.array(obj_info["cam_R_m2c"]).reshape((3,3))
    t = np.array(obj_info["cam_t_m2c"])/1000

    trans = np.eye(4)
    trans[0:3,0:3] = R
    trans[0:3,3]= t
    T = np.eye(4)
    T[1, 1] *= -1
    T[2, 2] *= -1

    scene = pyrender.Scene()
    scene.add(meshes[obj_id], pose=trans)
    cam = pyrender.IntrinsicsCamera(*cam_intr)
    scene.add(cam, pose=T)

    color, depth = r.render(scene)
    depth_img = depth
    
    depth_img[depth_img != 0] = 255 
    return  depth_img.astype(np.uint8)

def render_rgb(obj_info, cam_intr, r, meshes):
    obj_id = int(obj_info["obj_id"])

    R = np.array(obj_info["cam_R_m2c"]).reshape((3,3))
    t = np.array(obj_info["cam_t_m2c"])/1000

    trans = np.eye(4)
    trans[0:3,0:3] = R
    trans[0:3,3]= t
    T = np.eye(4)
    T[1, 1] *= -1
    T[2, 2] *= -1

    scene = pyrender.Scene(ambient_light=[1.,1.,1.,1.], bg_color=[0.,0.,0.])
    scene.add(meshes[obj_id], pose=trans)
    cam = pyrender.IntrinsicsCamera(*cam_intr)
    scene.add(cam, pose=T)

    #light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
    #                        innerConeAngle=np.pi/16.0,
    #                        outerConeAngle=np.pi/6.0)
    #scene.add(light, pose=T)
    
    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    return color

def calc_px_count_all(obj_info, cam_intr, r, meshes):
    obj_id = int(obj_info["obj_id"])
    R = np.array(obj_info["cam_R_m2c"]).reshape((3,3))
    t = np.array(obj_info["cam_t_m2c"])
    
    Top = np.eye(4)
    Top[1, 1] *= -1
    Top[2, 2] *= -1

    #z_straight = np.linalg.norm(t)
    t_list = np.array([[0, 0, t[2]]]).T
    t_list = t_list.flatten().tolist()
    T_2obj = lookAt(np.array(t).T, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    R_2obj = T_2obj[:3, :3]
    R = R_2obj @ R

    t = np.array(t_list).reshape((1,3))

    trans = np.eye(4)
    trans[0:3,0:3] = R
    trans[0:3,3]= t/1000

    scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02],
                        bg_color=[1.0, 1.0, 1.0])
    scene.add(meshes[obj_id], pose=trans)
    cam = pyrender.IntrinsicsCamera(*cam_intr)
    scene.add(cam, pose=Top)

    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
    scene.add(light)

    color, depth = r.render(scene)
    full_obj_depth_img = depth
    
    px_count_all = np.count_nonzero(full_obj_depth_img)
    return px_count_all

def calc_boundingbox_from_mask(mask):
    non_zero = np.nonzero(mask)

    if non_zero[0].size != 0:
        bb_xmin = np.nanmin(non_zero[1])
        bb_xmax = np.nanmax(non_zero[1])
        bb_ymin = np.nanmin(non_zero[0])
        bb_ymax = np.nanmax(non_zero[0])
        bbox = [int(bb_xmin), int(bb_ymin), int(bb_xmax - bb_xmin), int(bb_ymax - bb_ymin)]
    else:
        bbox = [int(0), int(0), int(0), int(0)]
    return bbox

def complete_dataset_to_bop(config, set_dir=None, q = None):

    root_dir = os.path.dirname(os.path.dirname(set_dir))
    mesh_dir = config["models_dir"]
    mesh_col_dir = config["models_col_dir"]

    cam_json = os.path.join(root_dir, "camera.json")

    mask_path = os.path.join(set_dir, 'mask')
    segmap_dir = os.path.join(set_dir, 'segmap')
    mask_col_dir = os.path.join(set_dir, 'mask_col')
    mask_visib_path = os.path.join(set_dir, 'mask_visib')
    gt_json = os.path.join(set_dir, "scene_gt.json")
    gt_info_json = os.path.join(set_dir, "scene_gt_info.json")

    segmap_filenames = sorted(os.listdir(segmap_dir))

    with open(cam_json, 'r') as file:
        cam_info = json.load(file)

    cam_intr = [cam_info['fx'], cam_info['fy'], cam_info['cx'], cam_info['cy']]   

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(mask_visib_path):
        os.makedirs(mask_visib_path)
    if not os.path.exists(mask_col_dir):
        os.makedirs(mask_col_dir)

    with open(gt_json, 'r') as file:
        scene_gt = json.load(file)

    meshes = {}
    meshes_col = {}
    r = pyrender.OffscreenRenderer(1280, 720)
    # for mesh_now in sorted(os.listdir(mesh_dir)):
    #     mesh_path_now = os.path.join(mesh_dir, mesh_now)
    #     mesh_path_col_now = os.path.join(mesh_col_dir, mesh_now)
    #     if mesh_now[-4:] != '.ply':
    #         continue
    #     mesh_id = int(mesh_now[4:-4])
    #     meshes[mesh_id] = pyrender.Mesh.from_trimesh(trimesh.load(mesh_path_now, file_type='ply'))
    #     meshes_col[mesh_id] = pyrender.Mesh.from_trimesh(trimesh.load(mesh_path_col_now, file_type='ply'))

    gt_infos_dict = {}

    if len(segmap_filenames) != 1000:
        return

    for img_num, objs_info in scene_gt.items():
        print(img_num)
        for item in objs_info:
            obj_id = item['obj_id']
            filename = "obj_{:06}.ply".format(obj_id)
            if obj_id not in meshes_col.keys():
                print("Add: obj_id {}".format(obj_id))
                #meshes[obj_id] = pyrender.Mesh.from_trimesh(trimesh.load(os.path.join(mesh_dir, filename), file_type='ply'))
                meshes_col[obj_id] = pyrender.Mesh.from_trimesh(trimesh.load(os.path.join(mesh_col_dir, filename), file_type='ply'))

        img_num = int(img_num)
        segmap_img = cv2.imread(os.path.join(segmap_dir, segmap_filenames[img_num]), cv2.IMREAD_UNCHANGED)
        gt_infos = []
        x = 0
        for obj_index, obj_info in enumerate(objs_info):
            obj_id = int(obj_info["obj_id"])
            #occlusion_mask = occlusion_mask_from_segmap(segmap_img, obj_id)
            #mask = render_mask(obj_info, cam_intr, r, meshes)
            #mask = mask.astype(np.uint8)
            mask_col = render_rgb(obj_info, cam_intr, r, meshes_col)
            #px_count_all = calc_px_count_all(obj_info, cam_intr, r, meshes)
            #px_count_valid = px_count_all
            #px_count_visib = np.count_nonzero(occlusion_mask)
            #if px_count_all > 0:
            #    visib_fract = 1.0 * px_count_visib/px_count_all
            #else: 
            #    visib_fract = 0

            #if visib_fract > 1:
            #    visib_fract = 1.0

            #bbox_obj = calc_boundingbox_from_mask(mask)
            #bbox_visib = calc_boundingbox_from_mask(occlusion_mask)

            mask_name = f"{img_num:06}_{obj_index:06}.png"

            #cv2.imwrite(os.path.join(mask_path,mask_name), mask)
            #cv2.imwrite(os.path.join(mask_visib_path,mask_name), occlusion_mask)
            cv2.imwrite(os.path.join(mask_col_dir,mask_name), mask_col)

            #gt_info_dict = {
            #    "bbox_obj": bbox_obj,
            #    "bbox_visib": bbox_visib,
            #    "px_count_all": px_count_all,
            #    "px_count_valid": px_count_valid,
            #    "px_count_visib": px_count_visib,
            #    "visib_fract": visib_fract
            #}

            #gt_infos.append(gt_info_dict)
        #gt_infos_dict[str(img_num)] = gt_infos
        #with open(gt_info_json, 'w') as f:
        #    json.dump(gt_infos_dict, f, indent=2)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__) #TODO
    #dirname = "/home/v4r/David/BlenderProc"

    filename = "/media/christian/master/dataset/deform_dataset/pyrender.txt"

    #read config
    with open(os.path.join(dirname, args.config_path), "r") as stream:
        config = yaml.safe_load(stream)
    root_path = os.path.join(config["output_dir"], "bop_data",config["dataset_name"])

    sets_path = os.path.join(root_path, 'train_pbr')

    sets = []
    f = open(filename, "r")
    for x in f:
        sets.append(x.split(",")[0])
    f.close()

    for set in sorted(os.listdir(sets_path)):
        print(sets_path)
        if set not in sets:
            print(set)
            start = timeit.default_timer()
            complete_dataset_to_bop(config, os.path.join(sets_path, set))
            stop = timeit.default_timer()
            print('Time: ', stop - start) 

            f = open(filename, "a")
            f.write("{},{}".format(set,stop - start))
            f.write("\n")
            f.close()

