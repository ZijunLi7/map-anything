import logging
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from argconf import argconf_parse
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
import matplotlib.cm
import matplotlib.pyplot as plt
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from mapanything.utils.wai.core import store_data
from mapanything.utils.wai.scene_frame import _filter_scenes
import kitti360scripts
from kitti360scripts.helpers.csHelpers import Rodrigues
from kitti360scripts.helpers import curlVelodyneData

logger = logging.getLogger(__name__)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def _load_poses_and_intrinsics(original_root, scene_name):
    pose_path = Path(original_root) / "data_poses" / scene_name / "cam0_to_world.txt"
    imutow_path = Path(original_root) / "data_poses" / scene_name / "poses.txt"
    intrinsics_path = Path(original_root) / "calibration" / "perspective.txt"

    poses = np.loadtxt(pose_path)
    frames = poses[:, 0].astype(int)
    poses = np.reshape(poses[:,1:],[-1,4,4])
    poses_cam0tow = {}
    for frame, pose in zip(frames, poses):
        poses_cam0tow[frame] = pose

    imutow = np.loadtxt(imutow_path)
    imu_frames = imutow[:,0].astype(int)
    imutow = np.reshape(imutow[:,1:],[-1,3,4])
    Tr_imutow= {}
    for frame, pose in zip(imu_frames, imutow):
        pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
        Tr_imutow[frame] = pose

    assert np.array_equal(frames, imu_frames), "Frame indices between cam0 and imu do not match"

    intrinsic_loaded = False
    width = -1
    height = -1
    with open(intrinsics_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_00:':
                R_rect = np.eye(4)
                R_rect[:3, :3] =np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == 'S_rect_00:':
                width = int(float(line[1]))
                height = int(float(line[2]))
    assert intrinsic_loaded, "Cannot find intrinsics for cam0"
    assert width > 0 and height > 0, "Cannot find image size for cam0"

    intrinsics_cam0 = {
        "K": K,
        "R_rect": R_rect,
        "width": width,
        "height": height
    }

    return poses_cam0tow, Tr_imutow, intrinsics_cam0

def _load_extrinsics(original_root):
    fileCameraToVelo = os.path.join(original_root, 'calibration', 'calib_cam_to_velo.txt')
    fid = open(fileCameraToVelo, 'r')
    last_row = np.array([0, 0, 0, 1]).reshape(1,4)
    TrCam0ToVelo = np.concatenate((np.loadtxt(fileCameraToVelo).reshape(3,4), last_row))
    fid.close()

    fileCameraToPose = os.path.join(original_root, 'calibration', 'calib_cam_to_pose.txt')
    fid = open(fileCameraToPose, 'r')
    TrCam0ToPose = np.concatenate((readVariable(fid, 'image_00', 3, 4), last_row))
    fid.close()

    TrVeloToCam0 = np.linalg.inv(TrCam0ToVelo)
    TrVeloToPose = TrCam0ToPose @ TrVeloToCam0

    return TrVeloToPose, TrVeloToCam0

def _load_kitti360_pointcloud(pt3d_path, frame_id, pose_imutow, TrVeloToPose, TrVeloToCam0):
    pt3d = np.fromfile(pt3d_path, dtype=np.float32).reshape(-1, 4)
    pt3d = pt3d.astype(np.float64)
    pt3d_curled = np.copy(pt3d)

    Tr_pose_pose = np.eye(4)
    if frame_id in pose_imutow.keys():
        if frame_id==1:
            if frame_id+1 in pose_imutow.keys():
                Tr_pose_pose = np.linalg.inv(pose_imutow[frame_id+1]) @ pose_imutow[frame_id]
        else:
            if frame_id-1 in pose_imutow.keys():
                Tr_pose_pose = np.linalg.inv(pose_imutow[frame_id]) @ pose_imutow[frame_id-1]
    Tr_delta = np.linalg.inv(TrVeloToPose) @ Tr_pose_pose @ TrVeloToPose

    r = Rodrigues(Tr_delta[0:3,0:3]).flatten()
    t = Tr_delta[0:3,3]
    pt3d_curled = curlVelodyneData.cCurlVelodyneData(pt3d, pt3d_curled, r, t)

    return pt3d_curled.astype(np.float32)

def camtoimage(pointsCam, K):
    ndim = pointsCam.ndim
    if ndim == 2:
        pointsCam = np.expand_dims(pointsCam, axis=0)  # (1, 3, N)
    points_proj = np.matmul(K[:3, :3].reshape([1, 3, 3]), pointsCam)  # (B, 3, N)
    depth = points_proj[:, 2, :]
    u = np.round(points_proj[:, 0, :] / (np.abs(depth) + 1e-6)).astype(np.int32)
    v = np.round(points_proj[:, 1, :] / (np.abs(depth) + 1e-6)).astype(np.int32)

    if ndim == 2:
        u = u[0]; v = v[0]; depth = depth[0];
    return u, v, depth

def process_kitti360_scene(cfg, scene_name, TrVeloToPose, TrVeloToCam0, visualize=True):

    scene_id =scene_name.split("_")[-2]

    scene_outpath = Path(cfg.root) / scene_name
    scene_outpath.mkdir(parents=True, exist_ok=True)
    image_dir = scene_outpath / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = scene_outpath / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)

    wai_frames = []

    pose_cam0tow, pose_imutow, intrinsics_cam0 = _load_poses_and_intrinsics(
        cfg.original_root, scene_name)
    TrVeloToRect = np.matmul(intrinsics_cam0["R_rect"], TrVeloToCam0)

    image_path = Path(cfg.original_root) / "data_2d_raw" / scene_name / "image_00" / "data_rect"
    pt3d_path = Path(cfg.original_root) / "data_3d_raw" / scene_name / "velodyne_points" / "data"

    for frame_id in pose_cam0tow.keys():
        image = image_path / ('%010d.png' % frame_id)
        pt3d = pt3d_path / ('%010d.bin' % frame_id)

        pt3d = _load_kitti360_pointcloud(pt3d, frame_id, pose_imutow, TrVeloToPose, TrVeloToCam0)
        pt3d[:, 3] = 1

        pts2Cam = np.matmul(TrVeloToRect, pt3d.T).T
        pointsCam = pts2Cam[:, :3]
        # project to image space
        u, v, depth = camtoimage(pointsCam.T, intrinsics_cam0["K"])
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        depthMap = np.zeros((intrinsics_cam0["height"], intrinsics_cam0["width"]))
        depthImage = np.zeros((intrinsics_cam0["height"], intrinsics_cam0["width"], 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<intrinsics_cam0["width"]), v>=0), v<intrinsics_cam0["height"])
        # visualize points within 30 meters
        # mask = np.logical_and(mask, depth>0)
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<30)
        depthMap[v[mask],u[mask]] = depth[mask]

        if visualize:
            cm = plt.get_cmap('jet')
            layout = (2,1)
            fig, axs = plt.subplots(*layout, figsize=(18,12))
            colorImage = np.array(Image.open(image)) / 255.
            depthImage = cm(depthMap/depthMap.max())[...,:3]
            colorImage[depthMap>0] = depthImage[depthMap>0]

            axs[0].imshow(depthMap, cmap='jet', interpolation='none')
            axs[0].title.set_text('Projected Depth')
            axs[0].axis('off')
            axs[1].imshow(colorImage)
            axs[1].title.set_text('Projected Depth Overlaid on Image')
            axs[1].axis('off')
            plt.suptitle('Sequence %s, Camera %s, Frame %010d' % (scene_id, '00', frame_id))
            os.makedirs('visualizations', exist_ok=True)
            visualize_path = 'visualizations/projected_depth_seq_%s_cam_%s_frame_%010d.png' % (scene_id, '00', frame_id)
            plt.savefig(visualize_path)
            print('Saved visualization to %s' % visualize_path)

def get_original_kitti360_names(cfg):
    # Get all scene names to process
    im_sensor_root = os.path.join(cfg.original_root, "data_2d_raw")
    original_scene_names = sorted(os.listdir(im_sensor_root))
    all_scene_names = _filter_scenes(
        cfg.root,
        original_scene_names,
        cfg.get("scene_filters")
    )
    return all_scene_names

if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion" / "kitti360.yaml")
    target_root = Path(cfg.root)
    target_root.mkdir(parents=True, exist_ok=True)
    TrVeloToPose, TrVeloToCam0 = _load_extrinsics(cfg.original_root)
    convert_scenes_wrapper(
        process_kitti360_scene,
        cfg,
        get_original_scene_names_func=get_original_kitti360_names,
        TrVeloToPose=TrVeloToPose,
        TrVeloToCam0=TrVeloToCam0
    )
