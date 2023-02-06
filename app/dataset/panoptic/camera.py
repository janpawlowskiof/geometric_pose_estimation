import json
from pathlib import Path
from typing import Dict, List

import torch
from matplotlib import pyplot as plt


def list_hd_cameras(clip_path: Path) -> List[str]:
    return [
        camera_path.stem
        for camera_path in clip_path.glob("hdImgs/??_??")
    ]


def get_camera_calibs(clip_path: Path) -> Dict[str, Dict]:
    calib_path = clip_path / f"calibration_{clip_path.stem}.json"
    with calib_path.open("r") as calib_file:
        calib = json.load(calib_file)

    return {
        camera_calib["name"]: camera_calib
        for camera_calib in calib["cameras"]
    }


def load_camera_matrix(camera_calib: Dict) -> torch.Tensor:
    """
    Return 3x4 camera matrix P that projects points from xyz foordinates to pixel space on camera
    This can be used to revert the process provided that the depth is known
    :param camera_calib: Camera dictionary as in panoptic dataset
    :return: 3x4 projection matrix
    """
    resolution = camera_calib["resolution"]
    R3 = torch.Tensor(camera_calib["R"])
    K3 = torch.Tensor(camera_calib["K"])
    t = torch.Tensor(camera_calib["t"]).squeeze(1)

    K4 = mat3_to_mat4(K3)
    R4 = mat3_to_mat4(R3)
    T4 = vec3_to_mat4(t)

    P4 = K4 @ T4 @ R4
    return P4[:3, :]


def mat3_to_mat4(mat3: torch.Tensor) -> torch.Tensor:
    mat4 = torch.eye(4)
    mat4[:3, :3] = mat3
    return mat4


def vec3_to_mat4(vec3: torch.Tensor) -> torch.Tensor:
    mat4 = torch.eye(4)
    mat4[:3, 3] = vec3
    return mat4


def display_points(x: torch.Tensor, P4: torch.Tensor, image: torch.Tensor):
    """
    Displays skeleton joints on image
    :param x: joints of shape (4, 19)
    :param P4: 4x4 camera projection matrix
    :param image: 3xHxW image to display joints on.
    :return: None
    """
    x_homo = P4 @ x
    x_proj = x_homo[:, :] / x_homo[2:3, :]

    body_edges = torch.Tensor([[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11], [11, 12]]) - 1

    plt.figure(figsize=(15, 15))
    plt.imshow(torch.permute(image, (1, 2, 0)))

    for edge in body_edges:
        plt.plot(x_proj[0, edge], x_proj[1, edge])
