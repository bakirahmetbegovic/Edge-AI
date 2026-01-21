from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import csv
import os
import numpy as np
import cv2
from ultralytics import YOLO

# Konfiguration
MODEL_PATH = "/home/orin/Desktop/yolo12n.pt"
DEVICE = "cpu"  # oder "cuda:0" für GPU
CONF = 0.6
IOU = 0.45
IMGSZ = 640

CAMERA_ID = 4  # 0 = Webcam, 4 = Intel RealSense D455f
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30

# CSV Output
OUT_CSV = "detections.csv"
DEDUP_DIST_PX = 50  # Pixel-Abstand für Deduplizierung

# Filter (optional)
TARGET_NAME_CONTAINS = ""  # z.B. "stop" oder "" für alle
TARGET_ID = None  # z.B. 0 oder None

WINDOW = "YOLO Detector (r=ROI, f=full, q=quit)"
# =========================================================


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, fw: int, fh: int) -> "ROI":
        x = max(0, min(self.x, fw - 1))
        y = max(0, min(self.y, fh - 1))
        w = max(1, min(self.w, fw - x))
        h = max(1, min(self.h, fh - y))
        return ROI(x, y, w, h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        r = self.clamp(fw, fh)
        return frame[r.y:r.y + r.h, r.x:r.x + r.w]

    def draw(self, frame: np.ndarray, color=(0, 255, 255)) -> None:
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.w, self.y + self.h), color, 2)


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    def center_px(self) -> Tuple[int, int]:
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)


class YOLODetector:
    def __init__(self, model_path: str, device: str, conf: float, 
                 iou: float, imgsz: int) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.names = getattr(self.model.model, "names", None) or \
                     getattr(self.model, "names", {})

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            source=bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        dets: List[Detection] = []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return dets

        r0 = res[0]
        xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            name = str(self.names.get(int(k), int(k)))
            dets.append(Detection(int(k), name, float(c), 
                                int(x1), int(y1), int(x2), int(y2)))
        return dets


class ROISearcher:
    def __init__(self, detector: YOLODetector, roi: ROI) -> None:
        self.detector = detector
        self.roi = roi

    def set_roi(self, roi: ROI) -> None:
        self.roi = roi

    def detect(self, frame: np.ndarray) -> List[Detection]:
        fh, fw = frame.shape[:2]
        r = self.roi.clamp(fw, fh)
        crop = r.crop(frame)
        dets_roi = self.detector.infer(crop)

        dets: List[Detection] = []
        for d in dets_roi:
            dets.append(
                Detection(
                    cls_id=d.cls_id,
                    cls_name=d.cls_name,
                    conf=d.conf,
                    x1=d.x1 + r.x, y1=d.y1 + r.y,
                    x2=d.x2 + r.x, y2=d.y2 + r.y,
                )
            )
        return dets


class ObjectMemory:
    """Deduplizierung basierend auf Pixel-Position"""
    def __init__(self, dist_thresh_px: float) -> None:
        self.dist_thresh = float(dist_thresh_px)
        self.points_by_class: dict = {}

    def load_from_csv(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cls_name = row.get("class", "")
                    if not cls_name:
                        continue
                    cx = row.get("px_cx")
                    cy = row.get("px_cy")
                    if cx and cy:
                        p = (float(cx), float(cy))
                        self.points_by_class.setdefault(cls_name, []).append(p)
        except Exception:
            pass

    def is_new(self, cls_name: str, center_px: Tuple[int, int]) -> bool:
        pts = self.points_by_class.get(cls_name, [])
        if not pts:
            return True
        cp = np.array(center_px, dtype=float)
        for q in pts:
            if float(np.linalg.norm(cp - np.array(q, dtype=float))) < self.dist_thresh:
                return False
        return True

    def add(self, cls_name: str, center_px: Tuple[int, int]) -> None:
        self.points_by_class.setdefault(cls_name, []).append(center_px)


class CSVLogger:
    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        self.fieldnames = [
            "timestamp", "class", "conf",
            "px_cx", "px_cy",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ]
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.out_path) and os.path.getsize(self.out_path) > 0:
            return
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        with open(self.out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()

    def append(self, det: Detection) -> None:
        cx, cy = det.center_px()
        row = {
            "timestamp": time.time(),
            "class": det.cls_name,
            "conf": float(det.conf),
            "px_cx": int(cx),
            "px_cy": int(cy),
            "bbox_x1": int(det.x1),
            "bbox_y1": int(det.y1),
            "bbox_x2": int(det.x2),
            "bbox_y2": int(det.y2),
        }
        with open(self.out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)


def select_roi(window: str, frame: np.ndarray, current: ROI) -> ROI:
    r = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        return current
    fh, fw = frame.shape[:2]
    return ROI(x, y, w, h).clamp(fw, fh)


def main() -> None:
    # YOLO laden
    try:
        detector = YOLODetector(MODEL_PATH, device=DEVICE, 
                               conf=CONF, iou=IOU, imgsz=IMGSZ)
    except Exception as e:
        print(f"Fehler beim Laden des YOLO-Modells: {e}")
        print(f"Prüfe MODEL_PATH: {MODEL_PATH}")
        return

    # Kamera öffnen
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    if not cap.isOpened():
        print(f"Fehler: Kamera {CAMERA_ID} konnte nicht geöffnet werden")
        return

    # Erstes Frame für Framegröße
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Kein Frame von Kamera")
        cap.release()
        return

    fh, fw = frame.shape[:2]
    print(f"Kamera: {fw}x{fh}")

    # Logger & Memory
    logger = CSVLogger(OUT_CSV)
    memory = ObjectMemory(dist_thresh_px=DEDUP_DIST_PX)
    memory.load_from_csv(OUT_CSV)

    # ROI: Start mit Fullframe
    roi = ROI(0, 0, fw, fh)
    searcher = ROISearcher(detector, roi)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    last_t = time.time()
    fps_s = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kein Frame mehr")
                break

            # FPS berechnen
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_s = 0.9 * fps_s + 0.1 * (1.0 / dt) if fps_s > 0 else (1.0 / dt)

            # Detektionen
            dets = searcher.detect(frame)
            saved_now = 0

            for d in dets:
                # Filter
                if TARGET_ID is not None and d.cls_id != TARGET_ID:
                    continue
                if TARGET_NAME_CONTAINS and \
                   (TARGET_NAME_CONTAINS.lower() not in d.cls_name.lower()):
                    continue

                # Deduplizierung
                center = d.center_px()
                if memory.is_new(d.cls_name, center):
                    memory.add(d.cls_name, center)
                    logger.append(d)
                    saved_now += 1

            # Visualisierung
            searcher.roi.draw(frame)
            for d in dets:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                cx, cy = d.center_px()
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}", 
                           (d.x1, max(18, d.y1 - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            total_saved = sum(len(v) for v in memory.points_by_class.values())
            cv2.putText(frame, 
                       f"FPS:{fps_s:.1f} | Gesamt:{total_saved} | Neu:{saved_now}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("f"):
                fh, fw = frame.shape[:2]
                searcher.set_roi(ROI(0, 0, fw, fh))
                print("ROI: Fullframe")
            elif key == ord("r"):
                searcher.set_roi(select_roi(WINDOW, frame, searcher.roi))
                print(f"ROI: {searcher.roi}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDetektionen gespeichert in: {OUT_CSV}")


if __name__ == "__main__":
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    main()