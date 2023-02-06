from pathlib import Path
from typing import Dict, List, Tuple

import torch.utils.data

from app.dataset.panoptic.camera import list_hd_cameras, get_camera_calibs, load_camera_matrix
from app.dataset.panoptic.images import list_frames, load_frame
from app.dataset.panoptic.joints import load_joints_with_single_person


class FlatPanopticClip(torch.utils.data.Dataset):
    def __init__(self, clip_path: Path, transforms):
        self.clip_path = clip_path
        self.transforms = transforms

        self.camera_matrices: Dict[str, torch.Tensor] = {
            camera_name: load_camera_matrix(camera_calib)
            for camera_name, camera_calib in get_camera_calibs(self.clip_path).items()
        }
        self.inv_camera_matrices: Dict[str, torch.Tensor] = {
            camera_name: torch.linalg.pinv(camera_matrix)
            for camera_name, camera_matrix in self.camera_matrices.items()
        }
        # frame index to index of person to their pose
        self.all_joints: Dict[int, Dict[int, torch.Tensor]] = load_joints_with_single_person(self.clip_path)
        # frame and camera indexes
        self.entries: List[Tuple[int, str]] = self.get_entries()

    def __getitem__(self, index):
        frame_index, camera_name = self.entries[index]
        joints: Dict[int, torch.Tensor] = self.all_joints[frame_index]
        assert len(joints) == 1, f"Flat panoptic clip should only use frames with one person but found {len(joints)}"
        joints: torch.Tensor = list(joints.values())[0]
        return {
            "joints": joints,
            "p": self.camera_matrices[camera_name],
            "inv_p": self.inv_camera_matrices[camera_name],
            "image": load_frame(self.clip_path, camera_name, frame_index, self.transforms)
        }

    def __len__(self) -> int:
        return len(self.entries)

    def get_entries(self) -> List[Tuple[int, str]]:
        result = []
        cameras: List[str] = list_hd_cameras(self.clip_path)
        for camera_name in cameras:
            for frame in list_frames(self.clip_path, camera_name):
                if frame in self.all_joints:
                    result.append((frame, camera_name))
        return result
