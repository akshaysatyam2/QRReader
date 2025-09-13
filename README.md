# QR Code Detection Project
This script for QR code detection and decoding pipeline using OpenCV. It has been designed to handle noisy, rotated, or low-contrast images. 

It leverages ArUco markers for QR localization/detection when available and falls back to full-image processing with multiple preprocessing variants when markers are absent. 

This pipeline outputs bounding boxes, decoded text, and visualizations, with optional debugged images for analysis.

## Setup

### Prerequesites

Python=>3.10
Virtual environment (recommended)

### Installation

Unzip the script folder.
Create and activate a virtual environment:
```
python -m venv .venv
.venv/bin/activate  # On Linux(Ubuntu) and MacOS: source .venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

requirements.txt:
```
opencv-contrib-python
numpy
packaging
```

## Running the Pipeline Script
Process images in the images/ directory and save results to outputs/:
```
python detect_qr.py
```
or 
```
python detect_qr.py --input_dir images --out_dir outputs
```

Enable debug mode to save intermediate images in debug/:
```
python detect_qr.py --debug
```
or
```
python detect_qr.py --input_dir images --out_dir outputs --debug
```

## Output
1. Console: For each image, prints <filename>, detected=<True|False>, decoded_text="<text>", bbox=[[x1,y1],...] (empty if undetected).
2. Visualizations: Saves <filename>_result.jpg in outputs/ with green polygon and text (if decoded) or red polygon (if detected only).
3. Debug: If --debug, saves intermediates (grayscale, ArUco input, markers, cropped, variants, enhanced ROI) to debug/<filename>/.
