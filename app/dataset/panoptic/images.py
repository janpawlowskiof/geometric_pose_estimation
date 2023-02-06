import re
from pathlib import Path
from typing import List

import torch
from PIL import Image
import torchvision


def load_frame(clip_path: Path, camera_name: str, frame: int) -> torch.Tensor:
    frame_path = clip_path / "hdImgs" / camera_name / f"{camera_name}_{frame:08d}.jpg"
    image = Image.open(frame_path)
    transform = torchvision.transforms.ToTensor()
    return transform(image)


def list_frames(clip_path: Path, camera_name: str) -> List[int]:
    camera_path = clip_path / "hdImgs" / camera_name
    frame_paths = list(camera_path.glob(f"{camera_name}_????????.jpg"))
    frames = [
        int(re.search(fr"{camera_name}_(?P<index>.*?).jpg", frame_path.name).group("index"))
        for frame_path in frame_paths
    ]
    return frames
