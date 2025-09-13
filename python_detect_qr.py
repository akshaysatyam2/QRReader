import argparse
from pathlib import Path

import cv2
import numpy as np
from packaging import version

def setup_aruco_detector(opencv_version, dictionary_type=cv2.aruco.DICT_6X6_250):
    """Initialize ArUco detector with tuned parameters for robust detection."""
    use_legacy_aruco = opencv_version < version.parse("4.7.0")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

    if use_legacy_aruco:
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minDistanceToBorder = 3
    params.minMarkerDistanceRate = 0.05
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.08
    params.perspectiveRemovePixelPerCell = 4
    params.maxErroneousBitsInBorderRate = 0.05
    params.errorCorrectionRate = 0.6

    detector = cv2.aruco.ArucoDetector(dictionary, params) if not use_legacy_aruco else None
    return detector, params, use_legacy_aruco, dictionary

def detect_aruco_markers(img, detector, params, use_legacy_aruco, dictionary, target_ids={43, 44, 101, 102}):
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

def crop_qr_region(img, marker_corners):
    """Crop image to QR region using marker corners, or return full image."""
    if len(marker_corners) == 4:
        all_points = np.vstack(marker_corners)
        min_pt = np.min(all_points, axis=0).astype(int)
        max_pt = np.max(all_points, axis=0).astype(int)
        margin = 30
        crop_y = slice(max(0, min_pt[1] - margin), min(img.shape[0], max_pt[1] + margin))
        crop_x = slice(max(0, min_pt[0] - margin), min(img.shape[1], max_pt[0] + margin))
        cropped = img[crop_y, crop_x]
        crop_offset = (crop_x.start, crop_y.start)
        return cropped, crop_offset, False
    return img, (0, 0), True

def preprocess_qr_variants(cropped, debug=False, debug_dir=None, filename=None):
    """Generate preprocessed image variants for QR detection (all BGR)."""
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

    if debug and debug_dir and filename:
        attempts_dir = debug_dir / filename / 'qr_attempts'
        attempts_dir.mkdir(exist_ok=True, parents=True)
        for name, img in variants:
            cv2.imwrite(str(attempts_dir / f"{name}.jpg"), img)

    return variants

def enhanced_decode(img, points, debug=False, debug_dir=None, filename=None):
    """Attempt decode on upscaled ROI with simple grayscale CLAHE enhancement."""
    pts_int = points.reshape(4, 2).astype(int)
    x_min, y_min = np.min(pts_int, axis=0)
    x_max, y_max = np.max(pts_int, axis=0)
    roi = img[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return ""

    roi_large = cv2.resize(roi, None, fx=2, fy=2)
    gray_roi = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray_roi)
    roi_enh = cv2.merge([enhanced_gray, enhanced_gray, enhanced_gray])
    roi_enh = np.ascontiguousarray(roi_enh)
    if roi_enh.dtype != np.uint8 or roi_enh.shape[2] != 3:
        roi_enh = cv2.convertScaleAbs(roi_enh)
    points_large = ((pts_int - [x_min, y_min]) * 2).astype(np.float32)

    try:
        qr_detector = cv2.QRCodeDetector()
        decoded_text, _ = qr_detector.decode(roi_enh, points_large)
        if debug and debug_dir and filename:
            cv2.imwrite(str(debug_dir / filename / 'enhanced_roi.jpg'), roi_enh)
        return decoded_text.strip()
    except cv2.error:
        print(f"Info: Enhanced decode failed due to processing error for {filename}.")
        return ""

def detect_and_decode_qr(img, needs_fallback, debug=False, debug_dir=None, filename=None):
    """Detect and decode QR code, with fallback preprocessing if needed."""
    qr_detector = cv2.QRCodeDetector()
    detect_success, points = qr_detector.detect(img)
    decoded_text = ""
    used_img = img

    if detect_success and points is not None:
        decoded_text, _ = qr_detector.decode(img, points)
        print(f"Info: QR detected in {filename} (decoded: {bool(decoded_text.strip())})")
    else:
        print(f"Info: No QR detected in {filename}")

    if needs_fallback and (not detect_success or not decoded_text.strip()):
        print(f"Info: Starting fallback preprocess attempts for {filename}...")
        variants = preprocess_qr_variants(img, debug, debug_dir, filename)
        for var_name, var_img in variants:
            if not detect_success:
                detect_success, points = qr_detector.detect(var_img)
                if detect_success:
                    print(f"Info: Detection succeeded on variant {var_name}.")
                    used_img = var_img
            if detect_success and points is not None and not decoded_text.strip():
                decoded_text, _ = qr_detector.decode(var_img, points)
                if decoded_text.strip():
                    print(f"Info: Decode succeeded on variant {var_name}.")
                    used_img = var_img

        if not detect_success:
            print(f"Info: All detection attempts failed for {filename}.")
        elif not decoded_text.strip():
            print(f"Info: QR detected but all decode attempts failed for {filename}.")

    if detect_success and points is not None and not decoded_text.strip():
        print(f"Info: Trying enhanced decode for {filename}...")
        enhanced_text = enhanced_decode(img, points, debug, debug_dir, filename)
        if enhanced_text:
            decoded_text = enhanced_text
            print("Info: Enhanced decode succeeded.")

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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    debug_dir = Path('debug')

    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    out_dir.mkdir(exist_ok=True)

    opencv_version = version.parse(cv2.__version__)
    print(f"Using OpenCV {cv2.__version__} (legacy ArUco: {opencv_version < version.parse('4.7.0')})")
    detector, detector_params, use_legacy_aruco, dictionary = setup_aruco_detector(opencv_version)

    for img_path in sorted(input_dir.glob('*.[jJ][pP][gG]')) + sorted(input_dir.glob('*.[pP][nN][gG]')):
        filename = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        marker_corners, gray, aruco_input, corners, ids = detect_aruco_markers(
            img, detector, detector_params, use_legacy_aruco, dictionary)
        if len(marker_corners) < 4:
            print(f"Warning: Only {len(marker_corners)} target markers found in {filename}, using full image.")
        else:
            print(f"Info: Cropped region for {filename} using 4 markers.")

        cropped, crop_offset, needs_fallback = crop_qr_region(img, marker_corners)

        detect_success, points, decoded_text, used_img = detect_and_decode_qr(
            cropped, needs_fallback, args.debug, debug_dir, filename)

        detected = bool(detect_success and points is not None)
        bbox = []
        if detected:
            points = points.reshape(4, 2).astype(int)
            points[:, 0] += crop_offset[0]
            points[:, 1] += crop_offset[1]
            bbox = points.tolist()

        print(f"{filename}, detected={str(detected)}, decoded_text=\"{decoded_text.strip()}\", bbox={bbox}")

        vis = visualize_result(img, detect_success, points, decoded_text)
        cv2.imwrite(str(out_dir / f"{filename}_result.jpg"), vis)

        if args.debug:
            save_debug_images(img, gray, aruco_input, corners, ids, cropped, used_img, 
                              debug_dir, filename)

if __name__ == "__main__":
    main()