# QR Code Detection Project

A robust QR code detection and decoding pipeline using OpenCV, designed to handle challenging real-world conditions including noisy, rotated  or low-contrast.

## Prerequisites

- Python >= 3.10
- Virtual environment (recommended)
- OpenCV with contrib modules

## Installation

1. Create and activate virtual environment
```
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/MacOS
source .venv/bin/activate
```

2. Install dependencies
```
pip install -r requirements.txt
```

requirements.txt:
```
opencv-contrib-python
numpy
packaging
```

## Usage

### Basic Usage
Process images in the default `images/` directory:
```
python python_detect_qr.py
```

### Custom Directories
Specify input and output directories:
```
python python_detect_qr.py --input_dir path_to_images --out_dir path_to_outputs
```

### Debug Mode
Save intermediate processing images for analysis:
```
python python_detect_qr.py --debug
```

### Advanced Options
```
python python_detect_qr.py --input_dir images --out_dir outputs --debug --marker_ids 43 44 101 102 --log_level DEBUG
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `images` | Directory containing input images |
| `--out_dir` | `outputs` | Directory to save output visualizations |
| `--debug` | `False` | Save intermediate debug images |
| `--marker_ids` | `[43, 44, 101, 102]` | ArUco marker IDs to detect |
| `--log_level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Output

### 1. Console Output
For each processed image, displays:
```
2024-01-15 10:30:45 - __main__ - INFO - image_name, detected=True, decoded_text="QR_CONTENT", bbox=[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

### 2. Visualization Files
Saves `<filename>_result.jpg` in outputs directory:
- Green polygon + text: Successfully decoded QR
- Red polygon: QR detected but not decoded
- No overlay: QR not detected

### 3. Debug Files (when --debug enabled)
Saves comprehensive debugging information:
- Grayscale conversion
- ArUco preprocessing
- Detected markers
- Cropped regions
- All preprocessing variants
- Enhanced ROI attempts

### 4. Processing Statistics
At completion, displays:
```
Total images: 6
Successful detections: 6/6 (100.0%)
Successful decodes: 6/6 (100.0%)
Average time per image: 0.44 seconds
```