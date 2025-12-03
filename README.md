# 🚀 Setup Rápido - Image Captioning

## 1️⃣ Instalar Dependencias
```bash
pip install -r requirements.txt
```

## 2️⃣ Descargar MS-COCO (20GB)
```bash
# Dar permisos al script
chmod +x download_coco.sh

# Ejecutar descarga (responder 'yes')
./download_coco.sh
```

## 3️⃣ Instalar unzip (si no lo tienes)
```bash
sudo apt-get update && sudo apt-get install -y unzip
```

## 4️⃣ Descomprimir COCO
```bash
cd ~/.cache/coco

# Descomprimir anotaciones (rápido)
unzip -q annotations_trainval2014.zip

# Descomprimir train (10-15 min)
unzip -q train2014.zip

# Descomprimir val (5-10 min)
unzip -q val2014.zip

cd -
```

## 5️⃣ Entrenar Modelo Base
```bash
cd src
python main.py
```

**Nota:** La primera vez extrae features (2-4 horas) y construye vocabulario (5-10 min)

## 6️⃣ Validación Externa (Opcional)
```bash
# 1. Agregar 10+ imágenes a:
imagenes_validacion/

# 2. Ejecutar
python validacion_externa.py
```

## 7️⃣ Variantes (Opcional)
```bash
# Variante 1: Fine-tuning
python src/variante_finetuning.py

# Variante 2: Beam Search
python src/variante_beam_search.py
```

---

## ⚡ Verificación Rápida

```bash
# Verificar COCO descargado
ls ~/.cache/coco/train2014/*.jpg | wc -l  # Debe ser ~73,571
ls ~/.cache/coco/val2014/*.jpg | wc -l    # Debe ser ~40,504

# Verificar GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 📁 Lo que se genera automáticamente:
- `features.pt` (~5-10GB) - Features pre-extraídas
- `vocabulary.pt` (~1-5MB) - Vocabulario
- `models/caption_model.pth` - Modelo entrenado

---

## 🐛 Problemas Comunes

**Out of Memory al extraer features:**
- Cierra otros programas
- O extrae features en partes (editar `extract_features.py`)

**El script se detiene:**
- Normal en extracción de features (toma horas)
- Revisa si `features.pt` existe
- Si existe pero está incompleto, bórralo y reinicia

---

**¡Listo! Con GPU el entrenamiento completo toma ~8-12 horas**
