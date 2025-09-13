import argparse
import logging
import time
from pathlib import Path
from typing import Set

import cv2
import numpy as np
from packaging import version

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with custom formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger(__name__)

def setup_aruco_detector(opencv_version, dictionary_type=cv2.aruco.DICT_5X5_1000):
    """Initialize ArUco detector with tuned parameters for robust detection."""
    use_legacy_aruco = opencv_version < version.parse("4.7.0")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

    if use_legacy_aruco:
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 2
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minDistanceToBorder = 3
    params.minMarkerDistanceRate = 0.05
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.08
    params.perspectiveRemovePixelPerCell = 4
    params.maxErroneousBitsInBorderRate = 0.05
    params.errorCorrectionRate = 0.6

    detector = cv2.aruco.ArucoDetector(dictionary, params) if not use_legacy_aruco else None
    return detector, params, use_legacy_aruco, dictionary

def detect_aruco_markers(img, detector, params, use_legacy_aruco, dictionary, target_ids: Set[int] = {43, 44, 101, 102}):
    """Detect ArUco markers in image with preprocessing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_input = cv2.GaussianBlur(gray, (5, 5), 0)
    aruco_input = cv2.adaptiveThreshold(aruco_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if use_legacy_aruco:
        corners, ids, _ = cv2.aruco.detectMarkers(aruco_input, dictionary, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(aruco_input)

    marker_corners = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in target_ids:
                marker_corners.append(corners[i][0])
    return marker_corners, gray, aruco_input, corners, ids

def crop_qr_region(img, marker_corners, adaptive_margin: bool = True):
    """Crop image to QR region using marker corners, or return full image."""
    if len(marker_corners) == 4:
        all_points = np.vstack(marker_corners)
        min_pt = np.min(all_points, axis=0).astype(int)
        max_pt = np.max(all_points, axis=0).astype(int)
        
        if adaptive_margin:
            area_width = max_pt[0] - min_pt[0]
            area_height = max_pt[1] - min_pt[1]
            margin = max(30, int(min(area_width, area_height) * 0.1))
        else:
            margin = 30
            
        crop_y = slice(max(0, min_pt[1] - margin), min(img.shape[0], max_pt[1] + margin))
        crop_x = slice(max(0, min_pt[0] - margin), min(img.shape[1], max_pt[0] + margin))
        cropped = img[crop_y, crop_x]
        crop_offset = (crop_x.start, crop_y.start)
        return cropped, crop_offset, False
    return img, (0, 0), True

def preprocess_qr_variants(cropped, debug=False, debug_dir=None, filename=None):
    """Generate enhanced preprocessed image variants for QR detection (all BGR)."""
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    variants = []

    var1 = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    variants.append(("1_grayscale_bgr", var1))

    adap = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    adap = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel)
    adap = cv2.morphologyEx(adap, cv2.MORPH_OPEN, kernel)
    var2 = cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR)
    variants.append(("2_adaptive_bgr", var2))

    _, otsu = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    var3 = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    variants.append(("3_otsu_bgr", var3))

    blurred = cv2.GaussianBlur(cropped, (9, 9), 10.0)
    var4 = cv2.addWeighted(cropped, 1.5, blurred, -0.5, 0)
    variants.append(("4_unsharp_bgr", var4))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_gray = clahe.apply(gray_crop)
    var5 = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    variants.append(("5_clahe_bgr", var5))

    bilateral = cv2.bilateralFilter(cropped, 9, 75, 75)
    variants.append(("6_bilateral_bgr", bilateral))

    adap_mean = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    var7 = cv2.cvtColor(adap_mean, cv2.COLOR_GRAY2BGR)
    variants.append(("7_adaptive_mean_bgr", var7))

    eq_gray = cv2.equalizeHist(gray_crop)
    var8 = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)
    variants.append(("8_hist_eq_bgr", var8))

    median = cv2.medianBlur(cropped, 5)
    variants.append(("9_median_bgr", median))

    kernel_grad = np.ones((5,5), np.uint8)
    gradient = cv2.morphologyEx(gray_crop, cv2.MORPH_GRADIENT, kernel_grad)
    _, gradient_thresh = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY_INV)
    var10 = cv2.cvtColor(gradient_thresh, cv2.COLOR_GRAY2BGR)
    variants.append(("10_morph_gradient_bgr", var10))

    if debug and debug_dir and filename:
        attempts_dir = debug_dir / filename / 'qr_attempts'
        attempts_dir.mkdir(exist_ok=True, parents=True)
        for name, img in variants:
            cv2.imwrite(str(attempts_dir / f"{name}.jpg"), img)

    return variants

def enhanced_decode(img, points, debug=False, debug_dir=None, filename=None):
    """Enhanced decode with multiple strategies."""
    try:
        pts_int = points.reshape(4, 2).astype(int)
        x_min, y_min = np.min(pts_int, axis=0)
        x_max, y_max = np.max(pts_int, axis=0)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)
        
        roi = img[y_min:y_max, x_min:x_max]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            logger.debug(f"ROI too small for {filename}")
            return ""

        scale_factors = [2.0, 3.0, 1.5]
        
        for scale in scale_factors:
            roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray_roi = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
            
            for clip_limit in [2.0, 3.0, 4.0]:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                enhanced_gray = clahe.apply(gray_roi)
                roi_enh = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                
                roi_enh = np.ascontiguousarray(roi_enh)
                if roi_enh.dtype != np.uint8:
                    roi_enh = cv2.convertScaleAbs(roi_enh)
                
                points_large = ((pts_int - [x_min, y_min]) * scale).astype(np.float32)
                
                try:
                    qr_detector = cv2.QRCodeDetector()
                    decoded_text, _ = qr_detector.decode(roi_enh, points_large)
                    if decoded_text and decoded_text.strip():
                        logger.info(f"Enhanced decode succeeded with scale={scale}, clip={clip_limit}")
                        if debug and debug_dir and filename:
                            cv2.imwrite(str(debug_dir / filename / f'enhanced_roi_scale{scale}_clip{clip_limit}.jpg'), roi_enh)
                        return decoded_text.strip()
                except Exception as e:
                    logger.debug(f"Enhanced decode attempt failed: {str(e)}")
                    continue
        
        for scale in [2.0, 3.0]:
            roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            qr_detector = cv2.QRCodeDetector()
            decoded_text, _, _ = qr_detector.detectAndDecode(roi_large)
            if decoded_text and decoded_text.strip():
                logger.info(f"Enhanced decode succeeded without points at scale={scale}")
                return decoded_text.strip()
                
    except Exception as e:
        logger.error(f"Enhanced decode failed for {filename}: {str(e)}")
    
    return ""

def multi_scale_detect(img, filename: str):
    """Try QR detection at multiple scales."""
    qr_detector = cv2.QRCodeDetector()
    scales = [1.0, 0.75, 1.25, 0.5, 1.5]
    
    for scale in scales:
        if scale != 1.0:
            scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scaled_img = img
            
        detect_success, points = qr_detector.detect(scaled_img)
        if detect_success and points is not None:
            decoded_text, _ = qr_detector.decode(scaled_img, points)
            if decoded_text and decoded_text.strip():
                if scale != 1.0:
                    points = points / scale
                logger.info(f"Multi-scale detection succeeded at scale={scale} for {filename}")
                return True, points, decoded_text.strip()
    
    return False, None, ""

def detect_and_decode_qr(img, needs_fallback, debug=False, debug_dir=None, filename=None):
    """Enhanced detect and decode QR code with multiple strategies."""
    qr_detector = cv2.QRCodeDetector()
    detect_success, points = qr_detector.detect(img)
    decoded_text = ""
    used_img = img

    if detect_success and points is not None:
        decoded_text, _ = qr_detector.decode(img, points)
        logger.info(f"QR detected in {filename} (decoded: {bool(decoded_text.strip())})")
    else:
        logger.info(f"No QR detected in {filename}, trying multi-scale detection...")
        detect_success, points, decoded_text = multi_scale_detect(img, filename)

    if needs_fallback and (not detect_success or not decoded_text.strip()):
        logger.info(f"Starting fallback preprocess attempts for {filename}...")
        variants = preprocess_qr_variants(img, debug, debug_dir, filename)
        
        for var_name, var_img in variants:
            if not detect_success:
                detect_success, points = qr_detector.detect(var_img)
                if detect_success:
                    logger.info(f"Detection succeeded on variant {var_name}")
                    used_img = var_img
                else:
                    temp_text, temp_points, _ = qr_detector.detectAndDecode(var_img)
                    if temp_text and temp_text.strip():
                        decoded_text = temp_text
                        points = temp_points
                        detect_success = True
                        logger.info(f"DetectAndDecode succeeded on variant {var_name}")
                        used_img = var_img
                        break
                        
            if detect_success and points is not None and not decoded_text.strip():
                decoded_text, _ = qr_detector.decode(var_img, points)
                if decoded_text.strip():
                    logger.info(f"Decode succeeded on variant {var_name}")
                    used_img = var_img
                    break

        if not detect_success:
            logger.warning(f"All detection attempts failed for {filename}")
        elif not decoded_text.strip():
            logger.warning(f"QR detected but all decode attempts failed for {filename}")

    if detect_success and points is not None and not decoded_text.strip():
        logger.info(f"Trying enhanced decode for {filename}...")
        enhanced_text = enhanced_decode(img, points, debug, debug_dir, filename)
        if enhanced_text:
            decoded_text = enhanced_text
            logger.info(f"Enhanced decode succeeded for {filename}")

    return detect_success, points, decoded_text, used_img

def visualize_result(img, detect_success, points, decoded_text):
    """Draw QR polygon and decoded text (if any) on the image."""
    vis = img.copy()
    if detect_success and points is not None:
        pts = points.reshape(-1, 1, 2).astype(int)
        color = (0, 255, 0) if decoded_text.strip() else (0, 0, 255)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
        if decoded_text.strip():
            rect = cv2.boundingRect(pts)
            cv2.putText(vis, decoded_text.strip(), (rect[0], max(0, rect[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis

def save_debug_images(img, gray, aruco_input, corners, ids, cropped, used_img, debug_dir, filename):
    """Save intermediate images for debugging."""
    debug_sub = debug_dir / filename
    debug_sub.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_sub / 'gray.jpg'), gray)
    cv2.imwrite(str(debug_sub / 'aruco_input.jpg'), aruco_input)
    img_markers = img.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)
    cv2.imwrite(str(debug_sub / 'detected_markers.jpg'), img_markers)
    cv2.imwrite(str(debug_sub / 'cropped.jpg'), cropped)
    cv2.imwrite(str(debug_sub / 'used_qr_input.jpg'), used_img)

def main():
    """Main function to process images and detect/decode QR codes."""
    parser = argparse.ArgumentParser(description='Detect and decode QR codes using OpenCV and ArUco markers.')
    parser.add_argument('--input_dir', type=str, default='images', help='Directory containing input images.')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Directory to save output visualizations.')
    parser.add_argument('--debug', action='store_true', help='Save intermediate debug images.')
    parser.add_argument('--marker_ids', type=int, nargs='+', default=[43, 44, 101, 102], 
                       help='ArUco marker IDs to detect')
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    debug_dir = Path('debug')

    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    out_dir.mkdir(exist_ok=True)

    opencv_version = version.parse(cv2.__version__)
    logger.info(f"Using OpenCV {cv2.__version__} (legacy ArUco: {opencv_version < version.parse('4.7.0')})")
    detector, detector_params, use_legacy_aruco, dictionary = setup_aruco_detector(opencv_version)

    target_ids = set(args.marker_ids)
    
    total_images = 0
    successful_detections = 0
    successful_decodes = 0
    start_time = time.time()

    for img_path in sorted(input_dir.glob('*.[jJ][pP][gG]')) + sorted(input_dir.glob('*.[pP][nN][gG]')):
        filename = img_path.stem
        total_images += 1
        
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not load {img_path}")
            continue

        marker_corners, gray, aruco_input, corners, ids = detect_aruco_markers(
            img, detector, detector_params, use_legacy_aruco, dictionary, target_ids)
        
        if len(marker_corners) < 4:
            logger.warning(f"Only {len(marker_corners)} target markers found in {filename}, using full image")
        else:
            logger.info(f"Cropped region for {filename} using 4 markers")

        cropped, crop_offset, needs_fallback = crop_qr_region(img, marker_corners)

        detect_success, points, decoded_text, used_img = detect_and_decode_qr(
            cropped, needs_fallback, args.debug, debug_dir, filename)

        detected = bool(detect_success and points is not None)
        bbox = []
        if detected:
            successful_detections += 1
            points = points.reshape(4, 2).astype(int)
            points[:, 0] += crop_offset[0]
            points[:, 1] += crop_offset[1]
            bbox = points.tolist()
            
            if decoded_text.strip():
                successful_decodes += 1

        logger.info(f"{filename}, detected={str(detected)}, decoded_text=\"{decoded_text.strip()}\", bbox={bbox}")

        vis = visualize_result(img, detect_success, points, decoded_text)
        cv2.imwrite(str(out_dir / f"{filename}_result.jpg"), vis)

        if args.debug:
            save_debug_images(img, gray, aruco_input, corners, ids, cropped, used_img, 
                              debug_dir, filename)

    elapsed_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Total images: {total_images}")
    logger.info(f"Successful detections: {successful_detections}/{total_images} ({successful_detections/total_images*100:.1f}%)")
    logger.info(f"Successful decodes: {successful_decodes}/{total_images} ({successful_decodes/total_images*100:.1f}%)")
    logger.info(f"Average time per image: {elapsed_time/total_images:.2f} seconds")

if __name__ == "__main__":
    main()