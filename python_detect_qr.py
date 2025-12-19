import argparse
import logging
import time
from typing import Set
from pathlib import Path
from packaging import version

import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode

def setup_logger(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("QR")

logger = setup_logger()

def setup_aruco():
    legacy = version.parse(cv2.__version__) < version.parse("4.7.0")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    params = cv2.aruco.DetectorParameters_create() if legacy else cv2.aruco.DetectorParameters()
    detector = None if legacy else cv2.aruco.ArucoDetector(dictionary, params)
    return detector, params, legacy, dictionary

def detect_aruco(img, detector, params, legacy, dictionary, target_ids: Set[int]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    if legacy:
        corners, ids, _ = cv2.aruco.detectMarkers(thresh, dictionary, parameters=params)
    else:
        corners, ids, _ = detector.detectMarkers(thresh)

    valid = []
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            if mid in target_ids:
                valid.append(corners[i][0])
    return valid

def crop_with_padding(img, corners):
    if len(corners) != 4:
        return img, (0, 0)

    pts = np.vstack(corners)
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    pad = int(min(x2 - x1, y2 - y1) * 0.2)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    return img[y1:y2, x1:x2], (x1, y1)

def zbar_decode_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = zbar_decode(gray)
    for r in results:
        try:
            return r.data.decode("utf-8")
        except Exception:
            continue
    return ""

def enhanced_decode(img, points):
    pts = points.reshape(4, 2).astype(int)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    pad = int(min(x2 - x1, y2 - y1) * 0.15)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)

    roi = img[y1:y2, x1:x2]
    if roi.size < 100:
        return ""

    qr = cv2.QRCodeDetector()

    for scale in [2, 3, 4, 5]:
        roi_s = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_s, cv2.COLOR_BGR2GRAY)

        for clip in [2.0, 3.0, 4.0]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
            txt, _, _ = qr.detectAndDecode(cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR))
            if txt.strip():
                return txt.strip()

        txt, _, _ = qr.detectAndDecode(roi_s)
        if txt.strip():
            return txt.strip()

    return zbar_decode_roi(roi)

def detect_and_decode_all(img):
    qr = cv2.QRCodeDetector()
    results = []

    ok, texts, points, _ = qr.detectAndDecodeMulti(img)
    if ok:
        for t, p in zip(texts, points):
            results.append({"points": p, "text": t.strip() if t else ""})

    if not results:
        for s in [0.75, 1.25, 1.5, 2.0]:
            simg = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            ok, texts, points, _ = qr.detectAndDecodeMulti(simg)
            if ok:
                for t, p in zip(texts, points):
                    results.append({"points": p / s, "text": t.strip() if t else ""})
                break

    for r in results:
        if not r["text"]:
            r["text"] = enhanced_decode(img, r["points"])

    return results

def visualize(img, qrs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    pad = 20

    vis = img.copy()
    labels = []

    for i, r in enumerate(qrs):
        pts = r["points"].reshape(-1, 1, 2).astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        if r["text"]:
            labels.append(f"QR #{i+1}: {r['text']}")

    if not labels:
        return vis

    widths, heights = [], []
    for l in labels:
        (w, h), b = cv2.getTextSize(l, font, scale, thickness)
        widths.append(w)
        heights.append(h + b + 8)

    text_w, text_h = max(widths), sum(heights)

    new_w = max(vis.shape[1], text_w + pad * 2)
    new_h = vis.shape[0] + text_h + pad * 2

    canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    x_off = (new_w - vis.shape[1]) // 2
    y_off = text_h + pad * 2

    canvas[y_off:y_off + vis.shape[0], x_off:x_off + vis.shape[1]] = vis

    y = pad + 20
    x = (new_w - text_w) // 2

    for line in labels:
        cv2.putText(canvas, line, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += 28

    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="Pictures")
    ap.add_argument("--out_dir", default="Output")
    ap.add_argument("--marker_ids", nargs="+", type=int, default=[43, 44, 101, 102])
    args = ap.parse_args()

    detector, params, legacy, dictionary = setup_aruco()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    total = decoded = 0
    start = time.time()

    for p in sorted(in_dir.glob("*.[jp][pn]g")):
        img = cv2.imread(str(p))
        if img is None:
            continue

        corners = detect_aruco(img, detector, params, legacy, dictionary, set(args.marker_ids))
        cropped, offset = crop_with_padding(img, corners)
        qrs = detect_and_decode_all(cropped)

        for r in qrs:
            r["points"][:, 0] += offset[0]
            r["points"][:, 1] += offset[1]
            total += 1
            if r["text"]:
                decoded += 1

        vis = visualize(img, qrs)
        cv2.imwrite(str(out_dir / f"{p.stem}_result.jpg"), vis)

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"Total QR found : {total}")
    logger.info(f"Successfully read : {decoded}")
    logger.info(f"Time taken : {elapsed:.2f}s")
    logger.info(f"Average time/img : {elapsed / max(1, total):.2f}s")

if __name__ == "__main__":
    main()
