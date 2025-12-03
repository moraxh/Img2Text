#!/bin/bash
# Script para descargar MS-COCO 2014 dataset

COCO_DIR="$HOME/.cache/coco"

echo "============================================"
echo "MS-COCO 2014 Dataset Downloader"
echo "============================================"
echo ""
echo "Este script descargará:"
echo "- Training images (13GB)"
echo "- Validation images (6GB)"
echo "- Annotations (241MB)"
echo ""
echo "Directorio de instalación: $COCO_DIR"
echo ""
read -p "¿Continuar? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Descarga cancelada."
    exit 0
fi

# Crear directorio
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

# Descargar Training images
echo ""
echo "Descargando training images (13GB)..."
if [ ! -f "train2014.zip" ]; then
    wget http://images.cocodataset.org/zips/train2014.zip
else
    echo "train2014.zip ya existe, saltando descarga."
fi

# Descargar Validation images
echo ""
echo "Descargando validation images (6GB)..."
if [ ! -f "val2014.zip" ]; then
    wget http://images.cocodataset.org/zips/val2014.zip
else
    echo "val2014.zip ya existe, saltando descarga."
fi

# Descargar Annotations
echo ""
echo "Descargando annotations (241MB)..."
if [ ! -f "annotations_trainval2014.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
else
    echo "annotations_trainval2014.zip ya existe, saltando descarga."
fi

# Descomprimir
echo ""
echo "Descomprimiendo archivos..."

if [ ! -d "train2014" ]; then
    echo "Descomprimiendo train2014.zip..."
    unzip -q train2014.zip
fi

if [ ! -d "val2014" ]; then
    echo "Descomprimiendo val2014.zip..."
    unzip -q val2014.zip
fi

if [ ! -d "annotations" ]; then
    echo "Descomprimiendo annotations..."
    unzip -q annotations_trainval2014.zip
fi

echo ""
echo "============================================"
echo "✓ Descarga completa!"
echo "============================================"
echo ""
echo "Estructura del dataset:"
ls -lh "$COCO_DIR"
echo ""
echo "Imágenes de entrenamiento: $(ls -1 $COCO_DIR/train2014/*.jpg | wc -l)"
echo "Imágenes de validación: $(ls -1 $COCO_DIR/val2014/*.jpg | wc -l)"
echo ""
echo "Puedes eliminar los archivos .zip para ahorrar espacio:"
echo "rm $COCO_DIR/*.zip"
echo ""
