# Pipeline ALPR: Detecção de Placas Veiculares & OCR

Pipeline de visão computacional para detecção e reconhecimento automático de placas veiculares (ALPR), com extração de caracteres via OCR e validação de padrões brasileiros (Mercosul e Tradicional).

---

## 🏗️ Como Funciona

```
Imagem do Veículo ➔ Detecção YOLO ➔ Enquadramento com Margem ➔ Filtros OpenCV (CLAHE/Bilateral) ➔ Motor OCR ➔ Validação Regex
```

1. **Detecção:** O modelo YOLO identifica as placas no quadro da imagem e devolve as coordenadas da *bounding box*.
2. **Recorte Dinâmico:** Uma margem percentual configurável é aplicada ao redor da placa para garantir que nenhum caractere seja cortado nas bordas.
3. **Pré-processamento com OpenCV:** Aplica redimensionamento bicúbico, filtro bilateral para redução de ruído, equalização local com CLAHE e limiarização adaptativa para gerar contraste ideal nos caracteres.
4. **Leitura OCR & Validação:** Os caracteres são extraídos via OCR (EasyOCR ou PyTesseract) e validados contra os padrões brasileiros:
   - **Mercosul:** Formato `ABC1D23` (`[A-Z]{3}[0-9][A-Z0-9][0-9]{2}`)
   - **Tradicional:** Formato `ABC1234` (`[A-Z]{3}[0-9]{4}`)

---

## 🚀 Como Executar

### 1. Instalação
```bash
git clone https://github.com/IgorBarduzzi/license-plate-detection-ocr.git
cd license-plate-detection-ocr

python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Inferência
```bash
python inference_ocr.py --model models/best_float16.tflite --source test_images/exemplo.jpg --outdir ocr_debug
```

Os recortes de cada placa, as imagens binarizadas e um arquivo JSON estruturado com as placas lidas serão salvos no diretório de saída informado.

### 3. Treinamento
Para treinar o detector com um dataset próprio anotado no formato Pascal VOC:
```bash
python train_plate_inplace.py
```

---

## 📁 Estrutura do Repositório

```
├── inference_ocr.py       # Pipeline de inferência YOLO, filtros OpenCV e leitura OCR
├── train_plate_inplace.py # Script de treinamento com TensorFlow Lite Model Maker
├── requirements.txt       # Dependências do projeto
└── README.md
```

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
