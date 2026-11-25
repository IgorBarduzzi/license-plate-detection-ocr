# Detecção e OCR de Placas Veiculares

Pipeline em Python para localizar placas veiculares em imagens com YOLO, tratar o recorte com OpenCV e extrair o texto por OCR (EasyOCR com fallback para Tesseract), classificando o resultado nos padrões brasileiros Mercosul (`ABC1D23`) e tradicional (`ABC1234`).

## Etapas do pipeline

1. **Detecção:** inferência via `ultralytics.YOLO` (`.pt` ou `.tflite`).
2. **Recorte com margem:** expansão percentual da bounding box para evitar corte nos caracteres das bordas.
3. **Pré-processamento:** upscale bicúbico 3x, filtro bilateral, CLAHE, unsharp masking, binarização adaptativa gaussiana e abertura morfológica.
4. **OCR e validação:** leitura dos caracteres alfanuméricos e checagem por regex para distinguir padrão Mercosul e tradicional.

## Instalação

```bash
git clone https://github.com/IgorBarduzzi/license-plate-detection-ocr.git
cd license-plate-detection-ocr
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Inferência em imagem

```bash
python inference_ocr.py --model models/best_float16.tflite --source imagem.jpg --outdir saida
```

O script salva no diretório de saída a imagem anotada, o recorte de cada placa, a versão binarizada enviada ao OCR e um JSON com coordenadas, confiança e texto lido.

### Treinamento

Com um dataset anotado no formato YOLO (`data.yaml` apontando para `train`/`val`):

```bash
python train_plate.py --data dataset/data.yaml --epochs 50 --export-tflite
```

## Licença

MIT — veja [LICENSE](LICENSE).
