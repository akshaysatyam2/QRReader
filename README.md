# QR Code Detection Project

A robust QR code detection and decoding pipeline using OpenCV, designed to handle challenging real-world conditions including noisy, rotated, or low-contrast images.

## Prerequisites

- Python >= 3.10
- Virtual environment (recommended)
- OpenCV with contrib modules

## Installation

1. **Create and activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/MacOS
source .venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-contrib-python
numpy
packaging
```

## Project Structure

```
├── python_detect_qr.py        # Main QR detection script
├── image_processor.py         # Image processing techniques (10 variants)
├── README.md
├── requirements.txt
├── Pictures/                  # Input images directory
└── Output/                    # Results and visualizations
```

## Usage

### Basic QR Detection
```bash
python python_detect_qr.py
```
Process images in the default `images/` directory.

### Custom Directories
```bash
python python_detect_qr.py --input_dir Pictures --out_dir Output
```

### Image Processing
```bash
python image_processor.py
```
Processes `Pictures/Picture_1.png` and saves 10 processing technique variants to `Output/` directory.

### Debug Mode
```bash
python python_detect_qr.py --debug
```

### Advanced Options
```bash
python python_detect_qr.py --input_dir Pictures --out_dir Output --debug --marker_ids 43 44 101 102 --log_level DEBUG
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `images` | Directory containing input images |
| `--out_dir` | `Output` | Directory to save output visualizations |
| `--debug` | `False` | Save intermediate debug images |
| `--marker_ids` | `43, 44, 101, 102` | ArUco marker IDs to detect |
| `--log_level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Output

**QR Detection Results** (`<filename>_result.jpg`)
- 🟢 Green polygon + text: Successfully decoded QR
- 🔴 Red polygon: QR detected but not decoded
- ⚪ No overlay: QR not detected

**Processing Techniques** (from `image_processor.py`)
- Grayscale conversion
- Histogram equalization
- Adaptive thresholding
- Gaussian blur
- Median filtering
- Morphological operations
- Canny edge detection
- Bilateral filtering
- Contrast stretching
- CLAHE enhancement

**Processing Statistics**
```
Total images: 6
Successful detections: 6/6 (100.0%)
Successful decodes: 6/6 (100.0%)
Average time per image: 0.44 seconds
```

---

Further reading: [10 Image Processing Techniques for Computer Vision](https://akshaysatyam2.medium.com/10-image-processing-techniques-for-computer-vision-d3df124b803c)
