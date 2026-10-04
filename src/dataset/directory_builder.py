"""
Directory Builder module for the dataset.
"""
import os
from pathlib import Path
from datetime import datetime

class DatasetDirectoryBuilder:
    """Creates and enforces a canonical output folder structure for the dataset."""
    
    def __init__(self, root: str, scene_prefix: str = "scene"):
        """
        Initializes the builder.
        
        Args:
            root (str): The root directory for the dataset.
            scene_prefix (str): Prefix appended to scene filenames.
        """
        self.root = Path(root)
        self.scene_prefix = scene_prefix
        self.folders = [
            "images",
            "depth",
            "masks",
            "labels/coco",
            "labels/yolo",
            "labels/kitti",
            "metadata"
        ]
        
    def build(self):
        """Creates all folders and generates the initial README.txt dataset card."""
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in self.folders:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            
        readme_path = self.root / "README.txt"
        if not readme_path.exists():
            self.update_readme(0, "N/A")

    def update_readme(self, total_scenes: int, camera_summary: str = "Standard Pinhole"):
        """
        Updates the dataset README card with the total scene count and parameters.
        
        Args:
            total_scenes (int): Total number of scenes exported.
            camera_summary (str): A string summarizing the camera/rendering parameters.
        """
        readme_path = self.root / "README.txt"
        content = f"""VisionDrive3D Synthetic Dataset
Dataset Root: {self.root.name}
Created/Last Updated: {datetime.now().isoformat()}
Total Scenes: {total_scenes}

Camera/Render Parameters: 
{camera_summary}

Folder Structure:
- images/       : RGB renders (.png)
- depth/        : Depth maps (.npy float32)
- masks/        : Segmentation masks (.png)
- labels/coco/  : COCO JSON annotations
- labels/yolo/  : YOLO .txt label files
- labels/kitti/ : KITTI 3D bbox label files (.txt)
- metadata/     : Per-scene JSON + global dataset_log.csv
"""
        readme_path.write_text(content, encoding='utf-8')

    def get_paths(self, scene_id: int) -> dict:
        """
        Returns a dictionary of absolute paths for a given scene export.
        
        Args:
            scene_id (int): Zero-indexed identifier for the scene.
            
        Returns:
            dict: Paths keyed by 'image', 'depth', 'mask', 'coco_json', 'yolo_txt', 'meta_json'.
        """
        base = f"{self.scene_prefix}_{scene_id:05d}"
        paths = {
            "image": self.root / "images" / f"{base}.png",
            "depth": self.root / "depth" / f"{base}.npy",
            "mask": self.root / "masks" / f"{base}.png",
            "coco_json": self.root / "labels" / "coco" / f"{base}.json",
            "yolo_txt": self.root / "labels" / "yolo" / f"{base}.txt",
            "kitti_txt": self.root / "labels" / "kitti" / f"{base}.txt",
            "meta_json": self.root / "metadata" / f"{base}.json",
        }
        return paths
