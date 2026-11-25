import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Treinamento de detector de placas com Ultralytics YOLO")
    ap.add_argument("--data", default="dataset/data.yaml", help="Arquivo data.yaml no formato YOLO")
    ap.add_argument("--weights", default="yolov8n.pt", help="Pesos base para fine-tuning")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--export-tflite", action="store_true", help="Exporta o melhor checkpoint para TFLite float16")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="plate_detector",
    )

    if args.export_tflite:
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        best_model = YOLO(str(best_pt))
        best_model.export(format="tflite", half=True, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
