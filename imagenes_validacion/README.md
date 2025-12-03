# Instrucciones para Validación Externa

Para cumplir con el requisito de validación externa, debes:

## 1. Agregar imágenes a la carpeta `imagenes_validacion/`

- Agrega **al menos 10 imágenes** que NO pertenezcan al dataset MS-COCO
- Incluye variedad: paisajes, personas, interiores, objetos, animales, etc.
- Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

## 2. Ejecutar el script de validación

```bash
python validacion_externa.py
```

## 3. Resultados

El script generará:

- **`resultados_validacion/captions.txt`**: Archivo de texto con todos los captions generados
- **`resultados_validacion/validation_results.png`**: Visualización en grid de imágenes con sus captions

## 4. Análisis Cualitativo

Analiza los resultados considerando:

- ¿El modelo detecta los objetos principales?
- ¿Hay alucinaciones (objetos mencionados que no están)?
- ¿La gramática es correcta?
- ¿Las descripciones son coherentes?

Incluye estos resultados en tu informe para demostrar la capacidad de generalización del modelo.
