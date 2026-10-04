# VisionDrive3D

**A 3D Rendering Pipeline for Synthetic Dataset Generation**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Visit%20Project-blue?logo=github)](https://benbunbo.github.io/VisionDrive3D/)

---

## 📋 Project Overview

**VisionDrive3D** is a Computer Graphics project that builds a comprehensive **3D rendering pipeline** for generating **synthetic images and annotated datasets**. The system is designed for AI training and evaluation, supporting various computer vision tasks including 2D/3D object detection, segmentation, depth estimation, and autonomous driving applications.

### Key Features

- **3D Rendering Engine**: Built with OpenGL 3.3 core for real-time rendering
- **Camera Management**: nuScenes-style surround camera rig mounted on an ego vehicle + free-flying camera
- **Ego-vehicle Dataset View**: headless mode auto-attaches to the **CAM_FRONT** of an available traffic car (1600×900), ego vehicle excluded from labels (nuScenes/KITTI convention)
- **Traffic Simulation**: Waypoint routing with Bezier turns, intersection control, collision avoidance, object pooling
- **Synthetic Dataset Generation**: Batch rendering with automatic annotation export (**COCO**, **YOLO**, **KITTI 3D** formats)
- **Multi-format Export**: RGB images, metric depth maps (float32 `.npy`), instance masks, KITTI-style 3D bounding boxes, and rich metadata
- **Multiview Export** (`--multiview`): render all 7 surround cameras at native resolution per frame
- **Headless Mode**: Offscreen FBO rendering for server-side generation (no visible window)
- **Performance**: Async PBO readbacks (single GPU sync per frame), binary mesh cache (startup ≈ 9s warm), decimated heavy assets
- **Real-time Preview**: Interactive viewer with RGB / Mask / Depth render modes

### Supported Output Formats

- **RGB Images**: High-quality synthetic photographs (PNG)
- **Depth Maps**: Per-pixel **metric depth** in meters (float32 NumPy `.npy`, linearized from the real depth buffer)
- **Instance Masks**: Per-object segmentation masks (PNG)
- **Annotations**:
  - COCO JSON format (2D boxes)
  - YOLO text format (2D boxes)
  - **KITTI text format** (2D + 3D boxes: `type truncated occluded alpha bbox h w l location rotation_y`)
- **Metadata**: Camera intrinsics/extrinsics, object poses, KITTI-style `bbox_3d` (`h, w, l, location, rotation_y, corners_world`)

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+ (3.11+ recommended; on 3.10 `imgui-bundle` is pinned to 1.5.2 — see Troubleshooting)
- pip + venv
- An OpenGL 3.3+ capable GPU/driver (headless servers: X display or Xvfb required)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ben-cp/VisionDrive3D.git
cd VisionDrive3D
```

### Step 2: Create a Virtual Environment

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip & Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note:** `torch` and `ultralytics` are **commented out** in `requirements.txt` (only needed for AI demos). Install them manually if required; for GPU support, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Step 4: Verify Installation

```bash
python -c 'import OpenGL, glfw, numpy, cv2, imgui_bundle, trimesh; print("All dependencies installed successfully!")'
```

---

## 🚀 Running the Project

> All commands are run from the **repository root**. The application entry point is the module `src.viewer`.

### 1. Interactive Viewer (GUI Mode)

```bash
python -m src.viewer
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `W/A/S/D` | Move camera (XZ plane) |
| `E` / `Q` (or `C`) | Move up / down |
| `Mouse` | Look around (free cam) |
| `TAB` | Switch active camera (free cam → surround cams) |
| `R` | Reset traffic routes (re-randomize) |
| `P` | Export current frame to the dataset |
| `G` | Batch-generate dataset scenes (re-randomize + export each) |
| `1` / `2` / `3` | Render mode: RGB / Mask / Depth |
| `T` | Toggle dimmed RGB lighting |
| `F` | Cycle fill mode (fill / wireframe / point) |
| `ESC` | Exit |

Useful flags (work in both modes): `--output <dir>` (default `./output_dataset`), `--multiview` (export all surround cameras instead of only the active one).

### 2. Batch Dataset Generation (Headless Mode)

```bash
python -m src.viewer --headless --frames 100 --output ./output_dataset
```

Behavior in headless mode:

- The active camera **auto-attaches to CAM_FRONT of an available traffic car** (ego front view, 1600×900); the ego vehicle is **excluded from labels**
- Cars spawn **once** from their traffic routes and then flow **continuously** — the dataset is a temporally coherent sequence (no per-frame re-randomization)
- Each captured frame advances the simulation by `--dt` seconds (**default 0.5 s**; use `0.016` for realtime, `1.0` for more diversity between frames)

Full options:

```bash
python -m src.viewer --headless \
    --frames 100 \                # number of scenes to generate
    --output ./output_dataset \   # dataset root (all artifacts go here)
    --dt 0.5 \                    # simulated seconds of traffic per frame
    --multiview                   # ALSO export all 7 surround cameras
```

On a headless server without a display:

```bash
xvfb-run -a python -m src.viewer --headless --frames 100 --output ./output_dataset
```

### 3. Scene Configuration & Assets

3D models live in `assets/`:

```
assets/
├── car0/  car1/  car2/     # vehicles: body_car.glb + 4 wheel GLBs + wheel_offsets.py
├── road_junction/          # base junction (OBJ + MTL + textures/)
└── scene/                  # street environment (OBJ + MTL + textures/)
```

Supported formats: `.obj` (with MTL/`map_Kd`) and `.glb` (glTF 2.0 binary, embedded or external textures).

> 💾 **Mesh cache**: on first load, parsed meshes are cached to `.mesh_cache/` (auto-generated, git-ignored). First startup ≈ 15 s, subsequent startups ≈ 9 s. Delete `.mesh_cache/` to force a re-parse (it also self-invalidates when a source file changes).

### 4. AI Demonstration

Run the AI inference demo on a generated dataset:

```bash
python -m src.ai_demo.ai_demo_runner --dataset ./output_dataset --scenes 5 --device cpu
```

### 5. Debug Tools

```bash
python -m src.tools.check_mask [mask_dir]   # inspect unique values in an instance-mask PNG
```

---

## 📁 Project Structure

```
BTL2/
├── src/
│   ├── viewer.py               # Main application & batch renderer (python -m src.viewer)
│   ├── core/
│   │   ├── entity.py           # Scene graph: Node/Entity/Scene, lane spawning
│   │   └── mesh.py             # OBJ/GLB loaders + binary mesh cache + GL building
│   ├── camera/
│   │   └── camera_suite.py     # Cameras, nuScenes surround rig, CameraManager
│   ├── scene/
│   │   ├── car.py              # Vehicle entity (body + wheels), route following
│   │   ├── traffic.py          # Traffic simulation (waypoints, intersections)
│   │   ├── scene_overlay.py    # Junction + buildings composition
│   │   └── api_scene_overlay.py# Junction OBJ loader / drawable API
│   ├── render/
│   │   └── renderers.py        # Render passes, FBO-based export, PBO readback
│   ├── annotation/
│   │   └── annotations.py      # 2D/3D bbox (OBB), KITTI labels, DatasetExporter
│   ├── dataset/                # Dataset manager, validators, metadata writers
│   ├── ai_demo/                # AI inference demonstrations
│   ├── vis/                    # Visualization tools
│   └── tools/
│       └── check_mask.py       # Mask debug tool
│
├── libs/                       # Core graphics libraries
│   ├── shader.py               # GLSL shader compiler
│   ├── buffer.py               # VAO/VBO/EBO management
│   ├── pbo.py                  # Async pixel readback (Pixel Buffer Objects)
│   ├── lighting.py             # Lighting & materials
│   ├── transform.py            # Matrix operations
│   └── camera.py               # Camera base helpers
│
├── shaders/                    # GLSL shader files
│   ├── phong.vert / phong.frag # Phong lighting (RGB pass)
│   ├── depth.vert / depth.frag # Depth visualization pass
│   └── mask.vert / mask.frag   # Instance/semantic masks
│
├── assets/                     # 3D models & textures
├── output_dataset/             # Generated synthetic data (default root)
├── .mesh_cache/                # Binary mesh cache (auto-generated)
└── requirements.txt
```

---

## 🔧 Dependencies

All dependencies are specified in `requirements.txt`. Key libraries:

- **PyOpenGL** (+accelerate): 3D graphics rendering
- **GLFW**: Window/context management and input
- **NumPy / OpenCV / Pillow**: Numerical computing & image I/O
- **pywavefront / trimesh**: OBJ & mesh processing
- **imgui-bundle** (pinned `==1.5.2` for Python 3.10): UI tooling
- **pandas / matplotlib**: Data handling & visualization
- *Optional (commented out)*: **torch**, **ultralytics** for AI demos

---

## 📊 Output Dataset Structure

```
output_dataset/
├── images/                   # RGB renders (active camera view)
├── depth/                    # Metric depth maps (.npy float32, meters)
├── masks/                    # Instance segmentation masks (.png)
├── labels/
│   ├── coco/                 # COCO JSON annotations (2D bbox)
│   ├── yolo/                 # YOLO .txt labels (2D bbox)
│   └── kitti/                # KITTI .txt labels (2D + 3D bbox)
├── metadata/
│   ├── scene_XXXXX.json      # Per-scene: camera params, objects, bbox_3d (KITTI-style)
│   └── dataset_log.csv       # Global dataset log
├── README.txt                # Dataset card
└── multiview/                # Only with --multiview
    ├── rgb/  mask/  depth/   # Per-camera files: <CAM_NAME>_NNNNNN.(png|npy)
    ├── labels/               # Per-camera YOLO labels
    ├── labels_kitti/         # Per-camera KITTI 3D labels
    └── scene_metadata_NNNNNN.json
```

**KITTI label line format** (15 fields, camera coordinates):

```
Car 0.00 0 -0.00 697.00 424.00 902.00 604.00 1.88 2.16 4.88 0.00 1.62 15.05 -0.00
# type truncated occluded alpha | bbox_x0 y0 x1 y1 | h w l | loc_x loc_y loc_z | rotation_y
```

- `location`: bottom-center of the 3D box in camera coords (x right, y down, z forward)
- `rotation_y`: object yaw relative to camera heading, radians `[-pi, pi]`
- `alpha`: observation angle = `rotation_y + atan2(loc_x, loc_z)`
- Full OBB available per object in `metadata/scene_XXXXX.json` → `bbox_3d.corners_world`

---

## 🎓 Architecture

The system follows a **top-down hierarchical rendering architecture**:

1. **Scene Graph** (`src/core/entity.py`): Nodes with TRS transforms, world-matrix hierarchy, entities with AABBs
2. **Render Manager** (`src/render/renderers.py`): Coordinates RGB/Mask render passes
3. **Export Pipeline**: offscreen FBO per camera resolution → async **PBO readbacks** (single `glFinish` per frame) → CPU labels (8-corner OBB projection) → writers (COCO/YOLO/KITTI)
4. **Drawables**: Objects that implement `setup()` and `draw()` methods

**Design Pattern**: All renderable objects must implement:

- `setup()` — Initialize OpenGL resources
- `draw(projection, view, model, shader)` — Render the object

---

## 📚 References

- [OpenGL Documentation](https://www.khronos.org/opengl/)
- [GLFW Documentation](https://www.glfw.org/)
- [KITTI Raw Data Format](http://www.cvlibs.net/datasets/kitti/)
- [nuScenes Dataset](https://www.nuscenes.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## 🔗 Project Links

- **GitHub Pages**: [https://benbunbo.github.io/VisionDrive3D/](https://benbunbo.github.io/VisionDrive3D/)
- **GitHub Repository**: [https://github.com/benbunbo/VisionDrive3D](https://github.com/benbunbo/VisionDrive3D)

---

## 📝 License

This project is part of a Computer Graphics coursework assignment (HK_252).

---

## 🤝 Support

For issues, questions, or suggestions, please refer to the GitHub Pages documentation or create an issue on the repository.

**Happy Rendering! 🎨🚗🎥**
