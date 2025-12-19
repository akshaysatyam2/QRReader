# QR Code Detection & Decoding Pipeline

A **robust, production-grade QR code detection and decoding system** designed to handle
real-world failures where OpenCV alone is insufficient.

This project combines:
- **OpenCV** → detection, geometry, cropping, visualization
- **ZXing/ZBar (via pyzbar)** → reliable decoding fallback
- **ArUco markers (optional)** → constrained search & noise reduction

This architecture is similar to what is used in **payment apps, scanners, and industrial CV systems**.

---

## Project Structure

```

├── python_detect_qr.py        # Main QR detection & decoding pipeline
├── image_processor.py         # 10 image preprocessing techniques (optional)
├── README.md
├── requirements.txt
├── Pictures/                 # Input images
├── Output/                   # Visualized results for image processor
└── Outputs/                  # Visualized results for QR Reader

````

---

## Requirements

- **Python ≥ 3.10**
- Linux / macOS / Windows  
  (Linux recommended for ZBar stability)

---

## Installation

### System dependency (ZBar)

**Linux (Ubuntu/Debian):**
```bash
sudo apt install libzbar0
````

**EndeavourOS / Arch Linux / Manjaro:**
```bash
sudo pacman -S zbar
````

**macOS (Homebrew):**

```bash
brew install zbar
```

**Windows:**

* Install ZBar binaries
* Or use WSL (recommended)

---

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic run (default directories)

```bash
python python_detect_qr.py
```

### Custom input/output

```bash
python python_detect_qr.py --input_dir Pictures --out_dir Output
```

### With ArUco marker constraints

```bash
python python_detect_qr.py --marker_ids 43 44 101 102
```

---

## How Decoding Works (Important)

1. **OpenCV detectAndDecodeMulti**
2. **Multi-scale retry**
3. **ROI-based enhanced decode (CLAHE + scaling)**
4. **ZXing/ZBar fallback (final authority)**

If a QR fails **after ZBar**, it is either:

* invalid
* decorative
* missing quiet zone
* non-standard

No further CV tricks will help.

---

## Output

For each input image:

* `<name>_result.jpg` is generated in `Output/`

Visualization includes:

* 🟢 Green polygon → decoded QR
* 🔴 Polygon only → detected but undecodable
* Text rendered on **expanded, centered canvas** (never clipped)

---

## Image Processing Utilities (Optional)
Update the picture name at the bottom of 'image_processor.py'
Run:
```bash
python image_processor.py
```

Generates:

* 10 preprocessing variants
* Side-by-side comparisons
* Collage output

Useful for **analysis, debugging, and documentation** — not required for decoding.

---

## Typical Performance

```
Total QR found    : 7
Successfully read : 7
Average time/img  : ~0.3s
```

Decode rate is limited by **QR validity**, not the pipeline.

---

## Design Philosophy

* Detection ≠ decoding
* Geometry first, decoding second
* Fallbacks are **explicit**, not magic
* OpenCV has limits — ZXing/ZBar exists for a reason

This is the **correct end-state** of a serious QR pipeline.

---

## Reference

Article:
[**10 Image Processing Techniques for Computer Vision**](https://akshaysatyam2.medium.com/10-image-processing-techniques-for-computer-vision-d3df124b803c)

---

> Note
> `pyzbar` **requires system-level ZBar** (`libzbar0` / `zbar`)
> It will NOT work with pip alone.

---