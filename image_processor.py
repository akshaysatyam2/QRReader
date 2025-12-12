
import os
from pathlib import Path

import cv2
import numpy as np

def add_watermark(img, text="@akshaysatyam2"):
    """
    Adds subtle watermark at bottom center.
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    color = (70, 70, 70)
    alpha = 0.4

    overlay = img.copy()
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = h - 25

    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def create_collage(original, processed_list, titles, output_path):
    """
    Creates a clean 3×4 grid: Original top-left, then all 10 processed versions.
    Text is centered with high-contrast, readable overlay.
    """
    cell_w, cell_h = 460, 460
    padding = 15
    grid = np.ones((3 * cell_h, 4 * cell_w, 3), dtype=np.uint8) * 250  # Light background

    def add_centered_text(img, text, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.1, thickness=3):
        h, w = img.shape[:2]
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = 70  # Fixed from top

        # Dark semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # White bold text
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Place Original (top-left)
    orig = cv2.resize(original, (cell_w, cell_h))
    add_centered_text(orig, "ORIGINAL", font_scale=1.6, thickness=4)
    grid[0:cell_h, 0:cell_w] = orig

    # Place all processed images
    for i, (proc, title) in enumerate(zip(processed_list, titles)):
        row = (i + 1) // 4
        col = (i + 1) % 4
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w

        resized = cv2.resize(proc, (cell_w, cell_h))
        add_centered_text(resized, title, font_scale=1.0, thickness=3)
        grid[y1:y2, x1:x2] = resized

    # Top title bar
    title_bar_h = 120
    title_bar = np.ones((title_bar_h, grid.shape[1], 3), dtype=np.uint8) * 255
    main_title = "10 Image Preprocessing Techniques for Computer Vision"
    text_size = cv2.getTextSize(main_title, cv2.FONT_HERSHEY_DUPLEX, 1.8, 5)[0]
    x = (title_bar.shape[1] - text_size[0]) // 2
    y = 85
    cv2.putText(title_bar, main_title, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.8, (30, 30, 80), 5, cv2.LINE_AA)

    # Final assembly
    final = np.vstack((title_bar, grid))
    final = add_watermark(final, "@akshaysatyam2")

    cv2.imwrite(output_path, final)
    print(f"Collage saved → {output_path}")

def side_by_side_with_title(original, processed, title):
    """
    Creates clean side-by-side with title on top and watermark.
    """
    h, w = original.shape[:2]
    if processed.shape[:2] != (h, w):
        processed = cv2.resize(processed, (w, h))

    orig = original.copy()
    proc = processed.copy()
    cv2.putText(orig, "ORIGINAL", (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(proc, title, (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)

    comparison = np.hstack((orig, proc))

    title_bar = np.ones((90, comparison.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_bar, title, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, (50, 50, 50), 3, cv2.LINE_AA)

    final = np.vstack((title_bar, comparison))

    final = add_watermark(final, "@akshaysatyam2")

    return final


def grayscale(img):
    """
    Converts image to grayscale.

    This preprocessor converts the input image to grayscale by removing color information, resulting in a single-channel image that emphasizes intensity variations.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale"

def adaptive_gaussian_threshold(img):
    """
    Applies adaptive Gaussian thresholding with morphological operations.

    This preprocessor uses adaptive Gaussian thresholding to binarize the image, followed by morphological closing and opening to reduce noise and enhance features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    k = np.ones((3,3), np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, k)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), "Adaptive Gaussian Threshold"

def otsu_threshold(img):
    """
    Applies Otsu's thresholding.

    Otsu's method automatically determines the optimal threshold value to separate foreground and background pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), "Otsu's Thresholding"

def unsharp_mask(img):
    """
    Applies unsharp masking for image sharpening.

    Unsharp masking enhances image sharpness by subtracting a blurred version of the image from the original, emphasizing edges and fine details.
    """
    blur = cv2.GaussianBlur(img, (9,9), 10)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0), "Unsharp Masking"

def clahe_enhancement(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE improves local contrast in images, making features more distinguishable, especially in areas with varying illumination.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR), "CLAHE Enhancement"

def bilateral_filtering(img):
    """
    Applies bilateral filtering for noise reduction while preserving edges.

    Bilateral filtering smooths images while maintaining edge sharpness, making it effective for noise reduction without blurring important features.
    """
    return cv2.bilateralFilter(img, 9, 75, 75), "Bilateral Filtering"

def adaptive_mean_threshold(img):
    """
    Applies adaptive mean thresholding.

    This preprocessor uses adaptive mean thresholding to binarize the image, which calculates the mean of the neighborhood pixels to determine the threshold for each pixel.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), "Adaptive Mean Threshold"

def histogram_equalization(img):
    """
    Applies histogram equalization.

    Histogram equalization enhances the contrast of the image by redistributing the intensity values, making features more distinguishable.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR), "Histogram Equalization"

def median_denoising(img):
    """
    Applies median filtering for noise reduction.

    Median filtering reduces noise in the image by replacing each pixel's value with the median value of its neighboring pixels, effectively removing salt-and-pepper noise.
    """
    return cv2.medianBlur(img, 5), "Median Denoising"

def morphological_edge(img):
    """
    Applies morphological gradient for edge enhancement.

    Morphological gradient highlights the edges in the image by calculating the difference between the dilation and erosion of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.ones((5,5), np.uint8)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    _, t = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), "Morphological Edge Enhancement"


PREPROCESSORS = [
    grayscale,
    adaptive_gaussian_threshold,
    otsu_threshold,
    unsharp_mask,
    clahe_enhancement,
    bilateral_filtering,
    adaptive_mean_threshold,
    histogram_equalization,
    median_denoising,
    morphological_edge,
]


def process_image(input_path="Pictures/Picture_1.png", output_dir="Output"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    base_name = Path(input_path).stem
    print(f"Processing: {input_path}")

    orig_watermarked = add_watermark(img.copy(), "@akshaysatyam2")
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_ORIGINAL.jpg"), orig_watermarked)

    processed_images = []
    titles = []

    for func in PREPROCESSORS:
        processed, title = func(img.copy())
        comparison = side_by_side_with_title(img, processed, title)

        safe_title = title.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
        filename = f"{base_name}__{safe_title}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, comparison)

        clean_name = f"{base_name}__{safe_title}_CLEAN.jpg"
        cv2.imwrite(os.path.join(output_dir, clean_name), processed)
        processed_images.append(processed)
        titles.append(title)

        print(f"Saved: {filename}")

    print(f"\nAll images saved to:\n   {os.path.abspath(output_dir)}")

    collage_path = os.path.join(output_dir, f"{base_name}_ALL_TECHNIQUES_COLLAGE.jpg")
    create_collage(img, processed_images, titles, collage_path)

    print(f"\nAll images saved to:\n {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    process_image(
        input_path="Pictures/Picture_1.png",
        output_dir="Output"
    )