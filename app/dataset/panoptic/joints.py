import json
import re
from pathlib import Path
from typing import List, Dict

import torch


def load_joints_with_single_person(clip_path: Path) -> Dict[int, Dict[int, torch.Tensor]]:
    result = {
        frame_index: joints_for_frame
        for frame_index, joints_for_frame
        in load_joints(clip_path).items()
        if len(joints_for_frame) == 1
    }

    return result


def load_joints(clip_path: Path) -> Dict[int, Dict[int, torch.Tensor]]:
    return {
        get_frame_index_from_path(frame_config_path): load_joints_for_frame(frame_config_path)
        for frame_config_path in (clip_path / "hdPose3d_stage1_coco19").glob("body3DScene_*.json")
    }


def get_frame_index_from_path(frame_config_path: Path) -> int:
    index = re.search(r"body3DScene_(?P<index>.*?).json", frame_config_path.name).group("index")
    return int(index)


def load_joints_for_frame(frame_config_path: Path) -> Dict[int, torch.Tensor]:
    with frame_config_path.open("r") as frame_config_file:
        frame_config = json.load(frame_config_file)
        return {
            int(body["id"]): reshape_joints(body["joints19"])
            for body in frame_config["bodies"]
        }


def reshape_joints(joints: List[float]) -> torch.Tensor:
    joints = torch.Tensor(joints)
    joints = joints.reshape([19, 4])
    # TODO: This is bad because I am overwriting confidence values
    joints[:, 3] = 1
    return joints.T
