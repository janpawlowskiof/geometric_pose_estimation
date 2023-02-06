import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data

from app.dataset.panoptic.camera import list_hd_cameras, get_camera_calibs, load_camera_matrix
from app.dataset.panoptic.images import list_frames
from app.dataset.panoptic.joints import load_joints


class PanopticClip(torch.utils.data.Dataset):
    def __init__(self, clip_path: Path, num_cameras: int = 3):
        raise NotImplementedError("This is not needed yet")
        self.clip_path = clip_path
        self.num_cameras = num_cameras

        self.camera_matrices: Dict[str, np.ndarray] = {
            camera_name: load_camera_matrix(camera_calib)
            for camera_name, camera_calib in get_camera_calibs(self.clip_path).items()
        }
        # frame index to index of person to their pose
        self.all_joints: Dict[int, Dict[int, np.ndarray]] = load_joints(self.clip_path)
        # frame index to list of cameras that have that frame
        self.frames: Dict[int, List[str]] = self.get_frames()
        self.dataset_index_to_frame_index: Dict[int, int] = dict(enumerate(self.frames.keys()))

    def __getitem__(self, index):
        index = self.dataset_index_to_frame_index[index]
        joints = self.all_joints[index]
        available_cameras = random.sample(self.frames[index], self.num_cameras)
        # matrices =

    def __len__(self) -> int:
        return len(self.frames)

    def get_frames(self) -> Dict[int, List[str]]:
        result = defaultdict(list)
        cameras: List[str] = list_hd_cameras(self.clip_path)
        for camera_name in cameras:
            for frame in list_frames(self.clip_path, camera_name):
                result[frame].append(camera_name)
        return self.filter_frames(result)

    def filter_frames(self, frames: Dict[int, List[str]]) -> Dict[int, List[str]]:
        return {
            frame_index: available_cameras
            for frame_index, available_cameras
            in frames.items()
            if len(available_cameras) >= self.num_cameras and frame_index in self.all_joints
        }
