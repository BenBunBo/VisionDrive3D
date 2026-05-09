# VisionDrive3D

**A 3D Rendering Pipeline for Synthetic Dataset Generation**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Visit%20Project-blue?logo=github)](https://ben-cp.github.io/VisionDrive3D/)

---

## 📋 Project Overview

**VisionDrive3D** is a Computer Graphics project that builds a comprehensive **3D rendering pipeline** for generating **synthetic images and annotated datasets**. The system is designed for AI training and evaluation, supporting various computer vision tasks including object detection, segmentation, and autonomous driving applications.

### Key Features

- **3D Rendering Engine**: Built with OpenGL for high-performance real-time rendering
- **Camera Management**: Multiple preset cameras (nuScenes surround cameras) plus free-flying camera
- **Traffic Simulation**: Dynamic traffic scenarios with multiple vehicle types
- **Synthetic Dataset Generation**: Batch rendering with automatic annotation export (COCO, YOLO formats)
- **Multi-format Export**: RGB images, depth maps, instance masks, and metadata
- **Headless Mode**: Support for server-side rendering without GUI
- **Real-time Preview**: Interactive viewer with scene management and overlay capabilities
- **Scene Composition**: Support for complex scenes with buildings, roads, and traffic

### Supported Output Formats

- **RGB Images**: High-quality synthetic photographs
- **Depth Maps**: Per-pixel depth information (NumPy format)
- **Instance Masks**: Per-object segmentation masks
- **Annotations**: 
  - COCO JSON format
  - YOLO text format
- **Metadata**: Scene configuration, camera parameters, and object information

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment manager (venv is recommended)

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

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing `torch` and `ultralytics` may take some time on first installation. For GPU support, refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Step 5: Verify Installation

```bash
python -c "import OpenGL; import glfw; import numpy; import torch; print('All dependencies installed successfully!')"
```

---

## 🚀 Running the Project

### 1. Interactive Viewer (GUI Mode)

Start the real-time viewer with the default scene:

```bash
python viewer.py
```

**Keyboard Controls:**
- `WASD` - Move camera (in free-cam mode)
- `Mouse` - Look around (disabled for trackball in other modes)
- `Tab` - Switch between camera modes
- `R` - Reset scene
- `ESC` - Exit

### 2. Batch Dataset Generation (Headless Mode)

Generate a synthetic dataset with multiple renders:

```bash
python viewer.py --headless --output-dir ./output_dataset --num-frames 100
```

This generates:
- RGB images in `output_dataset/images/`
- Depth maps in `output_dataset/depth/`
- Instance masks in `output_dataset/masks/`
- Annotations in `output_dataset/labels/` (COCO & YOLO formats)
- Metadata in `output_dataset/metadata/`

### 3. Scene Configuration & Assets

Place your 3D models in the `assets/` directory:

```
assets/
├── car0/
├── car1/
├── car2/
├── road_junction/
└── scene/
```

For each asset, prepare:
- `.obj` or `.mtl` files for models
- Texture files (PNG, JPG) in a `textures/` subdirectory

### 4. AI Demonstration

Run the AI inference demo to test object detection on rendered images:

```bash
python src/ai_demo/ai_demo_runner.py --input-dir output_dataset/images/ --model yolo26n.pt
```

---

## 📁 Project Structure

```
VisionDrive3D/BTL2/
├── viewer.py                 # Main application & batch renderer
├── entity.py                 # Scene graph & entity management
├── scene_overlay.py          # Scene rendering system
├── traffic.py                # Traffic simulation manager
├── car.py                    # Vehicle models & animation
├── renderers.py              # Rendering pipeline (Phong, depth, masks)
├── camera_suite.py           # Camera management system
├── mesh.py                   # 3D model loader (OBJ, MTL, textures)
├── annotations.py            # Dataset annotation & export
│
├── libs/                     # Core graphics libraries
│   ├── shader.py             # GLSL shader compiler
│   ├── buffer.py             # VAO/VBO/EBO management
│   ├── lighting.py           # Lighting & materials
│   ├── transform.py          # Matrix operations & cameras
│   └── camera.py             # Camera base classes
│
├── shaders/                  # GLSL shader files
│   ├── phong.vert / phong.frag      # Phong lighting
│   ├── depth.vert / depth.frag      # Depth rendering
│   └── mask.vert / mask.frag        # Segmentation masks
│
├── src/
│   ├── ai_demo/              # AI inference demonstrations
│   ├── dataset/              # Dataset utilities
│   └── vis/                  # Visualization tools
│
├── assets/                   # 3D models & textures
├── output_dataset/           # Generated synthetic data
├── outputs/                  # Temporary rendering outputs
└── requirements.txt          # Python dependencies
```

---

## 🔧 Dependencies

All dependencies are specified in `requirements.txt`. Key libraries:

- **OpenGL** (PyOpenGL): 3D graphics rendering
- **GLFW**: Window management and input handling
- **NumPy**: Numerical computing
- **OpenCV**: Image processing
- **PyTorch**: Deep learning framework (for AI tasks)
- **Ultralytics YOLO**: Object detection model
- **Trimesh**: 3D mesh processing
- **Pillow**: Image I/O
- **Matplotlib**: Data visualization

---

## 📊 Output Dataset Structure

```
output_dataset/
├── images/                   # RGB rendered images
├── depth/                    # Depth maps (.npy files)
├── masks/                    # Instance segmentation masks
├── labels/
│   ├── coco/                # COCO format annotations (.json)
│   └── yolo/                # YOLO format annotations (.txt)
├── metadata/                 # Scene metadata
│   ├── dataset_log.csv       # Rendering log
│   └── scene_XXXXX.json      # Per-frame metadata
└── README.txt                # Dataset information
```

---

## 🎓 Architecture

The system follows a **top-down hierarchical rendering architecture**:

1. **Scene Graph** (`entity.py`): Manages objects and their transformations
2. **Render Manager** (`renderers.py`): Coordinates multiple render passes
3. **Drawables**: Objects that implement `setup()` and `draw()` methods
4. **Shader Pipeline**: Multiple rendering modes (Phong, depth, masks)

**Design Pattern**: All renderable objects must implement:
- `setup()` - Initialize OpenGL resources
- `draw(projection, view, model)` - Render the object

---

## 🐛 Troubleshooting

### Import Errors
If you encounter import errors, ensure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### GLFW Window Not Opening
- Ensure your graphics drivers are up to date
- Check that OpenGL 3.3+ is supported on your system

### CUDA/GPU Issues
If you want GPU acceleration for PyTorch:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Issues
- Run in headless mode for faster batch processing
- Reduce image resolution if needed
- Check GPU usage with `nvidia-smi` (NVIDIA GPUs only)

---

## 📚 References

- [OpenGL Documentation](https://www.khronos.org/opengl/)
- [GLFW Documentation](https://www.glfw.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## 🔗 Project Links

- **GitHub Pages**: [https://ben-cp.github.io/VisionDrive3D/](https://ben-cp.github.io/VisionDrive3D/)
- **GitHub Repository**: [https://github.com/ben-cp/VisionDrive3D](https://github.com/ben-cp/VisionDrive3D)

---

## 📝 License

This project is part of a Computer Graphics coursework assignment (HK_252).

---

## 🤝 Support

For issues, questions, or suggestions, please refer to the GitHub Pages documentation or create an issue on the repository.

**Happy Rendering! 🎨🚗🎥**
