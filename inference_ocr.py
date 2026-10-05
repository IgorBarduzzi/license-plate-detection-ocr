import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def crop_with_margin(img_bgr, xyxy, margin=0.10):
    """Recorta a bbox (xyxy) com uma margem percentual."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_for_ocr(bgr):
    """Pré-processamento simples para ajudar OCR (não é OCR)."""
    # Upscale (ajuda OCR em placas pequenas)
    scale = 3
    bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Denoise + contraste local
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Nitidez leve
    sharp = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gray = cv2.addWeighted(gray, 1.6, sharp, -0.6, 0)

    # Binarização adaptativa
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    # Limpeza leve (remove ruído)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    return thr


def run_on_image(model, source_path: Path, out_dir: Path, imgsz: int, conf: float, iou: float, max_det: int, margin: float):
    img_bgr = cv2.imread(str(source_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {source_path}")

    # Ultralytics já devolve boxes em pixels no tamanho original da imagem
    res = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False
    )[0]

    # Imagem anotada (com boxes desenhadas)
    annotated = res.plot()  # BGR
    cv2.imwrite(str(out_dir / f"{source_path.stem}_annotated.jpg"), annotated)

    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
    cls_ids = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)

    dets = []
    for i, (xyxy, score, cls_id) in enumerate(zip(boxes, scores, cls_ids)):
        crop, xyxy_m = crop_with_margin(img_bgr, xyxy, margin=margin)
        if crop is None:
            continue

        crop_path = out_dir / f"{source_path.stem}_det{i}_score{score:.2f}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        proc = preprocess_for_ocr(crop)
        proc_path = out_dir / f"{source_path.stem}_det{i}_score{score:.2f}_proc.png"
        cv2.imwrite(str(proc_path), proc)

        dets.append({
            "i": int(i),
            "class_id": int(cls_id),
            "class_name": res.names.get(int(cls_id), str(cls_id)),
            "score": float(score),
            "xyxy_pixels_original": [int(x) for x in xyxy_m],
            "crop_file": str(crop_path.name),
            "proc_file": str(proc_path.name),
        })

    # Salva um JSON com coordenadas (pra integrar com app)
    meta = {
        "source": str(source_path.name),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "margin": margin,
        "detections": dets
    }
    with open(out_dir / f"{source_path.stem}_detections.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(dets)} detecções")
    print(f"Saída: {out_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test_images/3carros.jpg", help="Imagem para testar (jpg/png)")
    ap.add_argument("--model", default="models/best_float16.tflite",
                    help="Modelo (.tflite ou .pt)")
    ap.add_argument("--outdir", default="ocr_debug", help="Pasta de saída")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.12)
    args = ap.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    run_on_image(model, source_path, out_dir, args.imgsz, args.conf, args.iou, args.max_det, args.margin)


if __name__ == "__main__":
    main()
