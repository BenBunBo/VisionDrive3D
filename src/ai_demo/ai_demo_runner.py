"""
AI Demo Runner.
Evaluates the synthetic dataset using pre-trained off-the-shelf models.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from transformers import DPTForDepthEstimation, DPTFeatureExtractor
except ImportError:
    DPTForDepthEstimation = None
    DPTFeatureExtractor = None


class AIDemoRunner:
    """Runs AI models on synthetic dataset outputs to verify usability."""
    
    def __init__(self, dataset_root: str, device: str = "cpu"):
        """
        Initializes the AI Demo Runner.
        
        Args:
            dataset_root (str): The path to the dataset.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.root = Path(dataset_root)
        self.device = device
        
        # Setup output folders for AI results
        self.ai_out_dir = self.root / "ai_results"
        self.dirs = {
            "detection": self.ai_out_dir / "detection",
            "segmentation": self.ai_out_dir / "segmentation",
            "depth_pred": self.ai_out_dir / "depth_pred",
            "comparisons": self.ai_out_dir / "comparisons"  # Specifically ensuring comparison exists
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # Get list of images and other ground elements to process
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.depth_dir = self.root / "depth"
        
        if not self.images_dir.exists():
            print(f"Warning: Dataset images directory {self.images_dir} not found.")
            self.image_paths = []
        else:
            self.image_paths = sorted(list(self.images_dir.glob("*.png")))
            
    def run_all(self):
        """Runs Tasks A, B, C in sequence and prints a summary."""
        print(f"--- Running AI Demo on {len(self.image_paths)} scenes ---")
        
        if len(self.image_paths) == 0:
            print("No images found in dataset. Exiting.")
            return
            
        print("\\nTask A: YOLOv8 Object Detection...")
        self.run_detection()
        
        print("\\nTask B: YOLOv8-Seg Instance Segmentation...")
        self.run_segmentation()
        
        print("\\nTask C: Intel DPT Monocular Depth Estimation...")
        self.run_depth_estimation()
        
        print("\\nGenerating Comparison Grids...")
        for img_path in self.image_paths:
            try:
                # E.g. 'scene_00000' -> extract '00000'
                scene_id_str = img_path.stem.split('_')[-1]
                scene_id = int(scene_id_str)
                self.generate_comparison_grid(scene_id)
            except Exception as e:
                print(f"Failed to generate grid for {img_path.name}: {e}")
                
        print("\\n--- AI Demo Completed ---")
        
    def run_detection(self):
        """Task A: Runs YOLOv8 object detection."""
        if YOLO is None:
            print("ultralytics YOLO not installed. Skipping.")
            return

        try:
            model = YOLO("yolov8n.pt")
            model.to(self.device)
        except Exception as e:
            print(f"Failed to load YOLOv8n: {e}")
            return
            
        total_conf = 0.0
        count = 0
        
        for img_path in self.image_paths:
            results = model(str(img_path), verbose=False)
            res = results[0]
            
            # Save annotated output
            out_img = res.plot()
            cv2.imwrite(str(self.dirs["detection"] / img_path.name), out_img)
            
            # Save JSON
            scene_data = []
            for box in res.boxes:
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                bbox = box.xyxy[0].tolist()
                
                scene_data.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": bbox
                })
                total_conf += conf
                count += 1
                
            json_path = self.dirs["detection"] / f"{img_path.stem}.json"
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(scene_data, f, indent=4)
                
        mean_conf = total_conf / count if count > 0 else 0.0
        print(f"  > Detection Mean Confidence: {mean_conf:.3f}")
        
    def run_segmentation(self):
        """Task B: Runs YOLOv8n-seg for instance/semantic segmentation."""
        if YOLO is None:
            print("ultralytics YOLO not installed. Skipping.")
            return

        try:
            model = YOLO("yolov8n-seg.pt")
            model.to(self.device)
        except Exception as e:
            print(f"Failed to load YOLOv8n-seg: {e}")
            return
            
        total_iou = 0.0
        iou_count = 0
        
        for img_path in self.image_paths:
            results = model(str(img_path), verbose=False)
            res = results[0]
            
            # Save overlay image
            out_img = res.plot()
            cv2.imwrite(str(self.dirs["segmentation"] / img_path.name), out_img)
            
            # Calculate IoU if mask exists
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists() and res.masks is not None:
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                pred_masks = res.masks.data.cpu().numpy()
                
                if gt_mask is not None and len(pred_masks) > 0:
                    combined_pred = np.max(pred_masks, axis=0) # shape (H, W) or model output res
                    combined_pred = cv2.resize(combined_pred, (gt_mask.shape[1], gt_mask.shape[0]))
                    combined_pred = (combined_pred > 0.5).astype(np.uint8)
                    
                    gt_mask_binary = (gt_mask > 0).astype(np.uint8)
                    
                    intersection = np.logical_and(combined_pred, gt_mask_binary).sum()
                    union = np.logical_or(combined_pred, gt_mask_binary).sum()
                    if union > 0:
                        total_iou += intersection / union
                        iou_count += 1
                        
        mean_iou = total_iou / iou_count if iou_count > 0 else 0.0
        print(f"  > Segmentation Mean IoU: {mean_iou:.3f}")
        
    def run_depth_estimation(self):
        """Task C: Runs Monocular Depth Estimation using Intel DPT."""
        if DPTForDepthEstimation is None or DPTFeatureExtractor is None:
            print("transformers not installed. Skipping.")
            return
            
        try:
            model_name = "Intel/dpt-large"
            feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
            model = DPTForDepthEstimation.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"Failed to load DPT model: {e}")
            return
            
        total_rmse = 0.0
        rmse_count = 0
        
        from PIL import Image
        import matplotlib.cm
        viridis = matplotlib.cm.get_cmap('viridis')
        
        for img_path in self.image_paths:
            image = Image.open(str(img_path)).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            output = prediction.cpu().numpy()
            
            # Prevent division by zero
            out_min, out_max = output.min(), output.max()
            if out_max > out_min:
                normalized = (output - out_min) / (out_max - out_min)
            else:
                normalized = np.zeros_like(output)
                
            formatted = (normalized * 255).astype("uint8")
            colored_depth = (viridis(formatted) * 255).astype("uint8")[:, :, :3]
            colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR) # For OpenCV save
            
            cv2.imwrite(str(self.dirs["depth_pred"] / img_path.name), colored_depth)
            
            # Compare to GT Depth
            # For 0-padded ID length extraction: E.g., scene_00000.png -> scene_00000.npy 
            gt_depth_path = self.depth_dir / f"{img_path.stem}.npy"
            if gt_depth_path.exists():
                try:
                    gt_depth = np.load(str(gt_depth_path))
                    gt_min, gt_max = gt_depth.min(), gt_depth.max()
                    if gt_max > gt_min:
                        gt_norm = (gt_depth - gt_min) / (gt_max - gt_min)
                    else:
                        gt_norm = np.zeros_like(gt_depth)
                        
                    pred_norm = normalized
                    if pred_norm.shape != gt_norm.shape:
                        pred_norm = cv2.resize(pred_norm, (gt_norm.shape[1], gt_norm.shape[0]))
                        
                    rmse = np.sqrt(np.mean((pred_norm - gt_norm) ** 2))
                    total_rmse += rmse
                    rmse_count += 1
                except Exception as e:
                    print(f"Failed to calculate RMSE for {img_path.name}: {e}")
                    
        mean_rmse = total_rmse / rmse_count if rmse_count > 0 else 0.0
        print(f"  > Depth Estimation Mean RMSE: {mean_rmse:.3f}")
        
    def generate_comparison_grid(self, scene_id: int):
        """
        Creates a side-by-side subplot visualization matrix.
        
        Args:
            scene_id (int): Scene identifier.
        """
        # We find the file from image paths first by stemming
        stem = f"scene_{scene_id:05d}"
        
        img_p = self.images_dir / f"{stem}.png"
        gt_depth_p = self.depth_dir / f"{stem}.npy"
        gt_mask_p = self.masks_dir / f"{stem}.png"
        
        det_p = self.dirs["detection"] / f"{stem}.png"
        pred_depth_p = self.dirs["depth_pred"] / f"{stem}.png"
        seg_p = self.dirs["segmentation"] / f"{stem}.png"
        
        def safe_imread(path):
            if path.exists():
                return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        def safe_imread_gray(path):
            if path.exists():
                return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            return np.zeros((100, 100), dtype=np.uint8)
            
        img = safe_imread(img_p)
        det = safe_imread(det_p)
        seg = safe_imread(seg_p)
        pred_depth = safe_imread(pred_depth_p)
        gt_mask = safe_imread_gray(gt_mask_p)
        
        if gt_depth_p.exists():
            try:
                gt_depth_raw = np.load(str(gt_depth_p))
                gt_min, gt_max = gt_depth_raw.min(), gt_depth_raw.max()
                if gt_max > gt_min:
                     gt_norm = (gt_depth_raw - gt_min) / (gt_max - gt_min)
                else:
                     gt_norm = np.zeros_like(gt_depth_raw)
            except:
                gt_norm = np.zeros((100, 100))
        else:
            gt_norm = np.zeros((100, 100))
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Scene {scene_id} Output Validation", fontsize=16)
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_norm, cmap='viridis')
        axes[0, 1].set_title("GT Depth")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt_mask, cmap='gray')
        axes[0, 2].set_title("GT Mask")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(det)
        axes[1, 0].set_title("YOLO Detection")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_depth)
        axes[1, 1].set_title("Predicted Depth")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(seg)
        axes[1, 2].set_title("Seg Overlay")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        out_path = self.dirs["comparisons"] / f"{stem}_comparison.png"
        plt.savefig(str(out_path))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./output_dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenes", type=int, default=5, help="number of scenes to demo on")
    args = parser.parse_args()
    
    runner = AIDemoRunner(args.dataset, args.device)
    # The 'scenes' parameter here sets how many total renders to generate before demo 
    # but based on prompt we just use `run_all()` to process the built dataset folder
    runner.run_all()
