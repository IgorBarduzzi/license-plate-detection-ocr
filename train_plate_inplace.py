import os
from pathlib import Path
from PIL import Image
from lxml import etree

from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.object_detector import DataLoader

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def parse_voc_xml(xml_path: Path):
    """
    Retorna: labels(list[str]), bboxes(list[list[int]] no formato [xmin, ymin, xmax, ymax])
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    labels = []
    bboxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        labels.append(name)
        bboxes.append([xmin, ymin, xmax, ymax])

    return labels, bboxes

def build_dataloader_from_mixed_folder(folder: Path, allowed_labels: set[str]):
    """
    Lê imagens e xml na mesma pasta.
    Só inclui imagens que têm xml correspondente e têm ao menos 1 box com label permitido.
    """
    dl = DataLoader([], [])

    images = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    images.sort()

    kept = 0
    skipped = 0

    for img_path in images:
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            skipped += 1
            continue

        labels, bboxes = parse_voc_xml(xml_path)

        # filtra somente a classe que você quer
        filtered = [(l, b) for (l, b) in zip(labels, bboxes) if l in allowed_labels]
        if not filtered:
            skipped += 1
            continue

        labels_f, bboxes_f = zip(*filtered)

        # carrega dimensões da imagem
        with Image.open(img_path) as im:
            w, h = im.size

        # Model Maker espera dict com "annotations" no formato:
        # {"boxes": [[xmin,ymin,xmax,ymax], ...], "classes": ["placa", ...]}
        record = {
            "image_path": str(img_path),
            "annotations": {
                "boxes": [list(bb) for bb in bboxes_f],
                "classes": list(labels_f),
            },
            "image_shape": (h, w),  # (height, width)
        }

        dl._dataset.append(record)  # usa a estrutura interna do DataLoader
        kept += 1

    print(f"[{folder}] kept={kept} skipped={skipped} total_imgs={len(images)}")
    return dl

def main():
    base_dir = Path(__file__).resolve().parent
    train_dir = base_dir / "dataset" / "train"
    val_dir   = base_dir / "dataset" / "valid"
    test_dir  = base_dir / "dataset" / "test"

    # sua classe no XML é "placa"
    allowed = {"placa"}

    train_data = build_dataloader_from_mixed_folder(train_dir, allowed)
    val_data   = build_dataloader_from_mixed_folder(val_dir, allowed)
    test_data  = build_dataloader_from_mixed_folder(test_dir, allowed)

    spec = model_spec.get("efficientdet_lite0")  # realtime
    model = object_detector.create(
        train_data,
        model_spec=spec,
        validation_data=val_data,
        epochs=50,
        batch_size=8
    )

    print("== Avaliação ==")
    model.evaluate(test_data)

    print("== Exportando TFLite ==")
    model.export(export_dir=str(base_dir / "export_plate_model"))

if __name__ == "__main__":
    main()