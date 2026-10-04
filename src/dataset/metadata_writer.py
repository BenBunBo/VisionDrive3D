"""
Metadata Writer module for the dataset.
"""
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.dataset.directory_builder import DatasetDirectoryBuilder

def _to_serializable(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

class MetadataWriter:
    """Serializes per-scene metadata to JSON and maintains a global dataset CSV log."""
    
    def __init__(self, dataset_root: str):
        """
        Initializes the metadata writer.
        
        Args:
            dataset_root (str): The root directory for the dataset.
        """
        self.root = Path(dataset_root)
        self.metadata_dir = self.root / "metadata"
        self.csv_path = self.metadata_dir / "dataset_log.csv"
        self.csv_headers = [
            "scene_id", "timestamp", "num_objects", "num_visible",
            "camera_x", "camera_y", "camera_z", "depth_min", "depth_max",
            "image_path", "coco_json_path"
        ]
        
        self.builder = DatasetDirectoryBuilder(dataset_root)
        
        # Ensure metadata dir exists in case called directly without builder
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
                
    def write_scene(self, scene_id: int, camera_params: dict, objects: list[dict], render_config: dict, depth_map: np.ndarray):
        """
        Writes the per-scene JSON and appends a row to the global CSV log.
        
        Args:
            scene_id (int): Scene identifier.
            camera_params (dict): Dictionary with camera configurations.
            objects (list[dict]): List of object dictionaries.
            render_config (dict): General rendering configuration.
            depth_map (np.ndarray): The depth map to compute depth statistics.
        """
        paths = self.builder.get_paths(scene_id)
        timestamp = datetime.now().isoformat()
        
        num_objects = len(objects)
        num_visible = sum(1 for obj in objects if obj.get("visible", True))
        
        depth_min = float(np.nanmin(depth_map)) if depth_map is not None else 0.0
        depth_max = float(np.nanmax(depth_map)) if depth_map is not None else 0.0
        
        # Relative paths
        image_relative = Path(paths["image"]).relative_to(self.root).as_posix()
        coco_relative = Path(paths["coco_json"]).relative_to(self.root).as_posix()
        
        scene_meta = {
            "scene_id": scene_id,
            "timestamp": timestamp,
            "image_path": image_relative,
            "camera": camera_params,
            "objects": objects,
            "render_config": render_config,
            "stats": {
                "num_objects": num_objects,
                "num_visible": num_visible,
                "depth_min": depth_min,
                "depth_max": depth_max
            }
        }
        
        # Write JSON
        with open(paths["meta_json"], "w", encoding="utf-8") as f:
            json.dump(_to_serializable(scene_meta), f, indent=4)
            
        # Append to CSV
        camera_pos = camera_params.get("position", [0.0, 0.0, 0.0])
        csv_row = [
            scene_id,
            timestamp,
            num_objects,
            num_visible,
            camera_pos[0],
            camera_pos[1],
            camera_pos[2],
            depth_min,
            depth_max,
            image_relative,
            coco_relative
        ]
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
            
    def flush_csv(self):
        """
        Ensures CSV is safely written.
        Because Python's 'with open(...)' context manager handles closing and flushing automatically,
        this acts as an explicit API compatibility placeholder.
        """
        pass
