# 🚀 Guía de Configuración de GPU en Google Colab

## El Problema

Cuando intentas entrenar en Colab con `--device cuda`, obtienes:
```
RuntimeError: Found no NVIDIA driver on your system.
```

Esto ocurre incluso si Colab tiene GPU disponible, porque:

1. **PyTorch se instala con CUDA incorrecto** - Tu notebook usa `cu117` que no existe en Colab 2025
2. **No se verifica GPU antes de instalar** - No hay forma de saber si CUDA está disponible
3. **Orden de instalación** - Instalar PyTorch después de otras librerías puede causar conflictos

## Solución: Pasos en Orden Correcto

### ✅ PASO 1: Verificar GPU en Colab

Primero, **en Google Colab**:
1. Ve a `Runtime` → `Change runtime type`
2. Selecciona **GPU** (T4, V100, A100 según disponibilidad)
3. Haz clic en `Save`

Ahora ejecuta en Colab para verificar GPU:
```bash
!nvidia-smi
```

Deberías ver algo como:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17    CUDA Version: 12.0   |
| Tesla T4              ...
| Memory-Usage: 123MiB / 15109MiB
+-----------------------------------------------------------------------------+
```

### ✅ PASO 2: Instalar PyTorch Correcto

En Colab, ejecuta:

```bash
# Desinstalar cualquier versión anterior
python3.8 -m pip uninstall torch torchvision torchaudio -y -q

# Instalar PyTorch para CUDA 12.1 (compatible con Colab 2025)
python3.8 -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q
```

### ✅ PASO 3: Verificar que PyTorch Ve GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Deberías ver:
```
CUDA available: True
CUDA version: 12.1
GPU name: Tesla T4
```

### ✅ PASO 4: Entrenar con GPU

```bash
cd /content/ML_RL
python3.8 train.py --n-episodes 100 --print-every 25 --device cuda
```

## Comparación: CPU vs GPU

| Aspecto | CPU | GPU |
|--------|-----|-----|
| Tiempo por episodio | ~0.5s | ~0.05s |
| 1000 episodios | ~500s (8 min) | ~50s (1 min) |
| Modelo GPU requerido | N/A | T4, V100, A100 |
| Disponibilidad en Colab | ✅ Siempre | ⚠️ Limitada |

## Plan B: Si GPU No Funciona

Si después de todos los pasos PyTorch aún no ve GPU:

### Opción 1: Usar CPU
```bash
python3.8 train.py --n-episodes 50 --print-every 10 --device cpu
```
(Será más lento pero funciona)

### Opción 2: Reintentar
1. Runtime → Disconnect
2. Runtime → Reconnect
3. Ejecuta las celdas de nuevo desde el principio

### Opción 3: Cambiar de GPU
1. Runtime → Change runtime type
2. Cambia a V100 si T4 no funciona
3. Reintentar

### Opción 4: Ejecutar Localmente
Si Colab no coopera, entrena en tu máquina local con:
```bash
python train.py --n-episodes 100 --print-every 25 --device cpu
```
(O `cuda` si tienes GPU local)

## Resumen: Checklist Colab

- [ ] Cambiar a GPU en Colab runtime settings
- [ ] Ejecutar `!nvidia-smi` → Verificar que muestra GPU
- [ ] Ejecutar `!git pull` → Traer cambios del repo
- [ ] Ejecutar celda de verificación GPU → Ver que PyTorch detecta CUDA
- [ ] Ejecutar `train.py` con `--device cuda`

## Archivos Importantes

- **`train.py`**: Script de entrenamiento (ya compatible con GPU/CPU)
- **`agent.py`**: Agente RL que maneja dispositivos
- **`requirements.txt`**: Dependencias (PyTorch se instala por separado)

## Versiones Instaladas en Colab

```
Python: 3.8
PyTorch: 1.13+ con CUDA 12.1
Gym: 0.21.0
MuJoCo: 2.1
TorchVision, TorchAudio: Latest
```

## Preguntas Frecuentes

**P: ¿Por qué mi notebook original falla?**  
R: Usa `cu117` que no existe. Colab 2025 necesita `cu121` o `cu118`.

**P: ¿Puedo entrenar 100,000 episodios en Colab?**  
R: Con GPU T4 te toma ~1 hora. Con CPU serían ~10 horas (se desconecta antes).

**P: ¿Mi modelo se guarda?**  
R: Sí, en `model.mdl` y `model_episode_N.mdl`. Descarga antes de que expire la sesión.

**P: ¿Qué pasa si se desconecta Colab?**  
R: Pierdes el modelo. Guarda periódicamente en Google Drive:
```python
from google.colab import files
files.download('model.mdl')
```

## Contacto

Si aún tienes problemas, revisa:
1. Error exacto en la terminal
2. Output de `nvidia-smi`
3. Output de `torch.cuda.is_available()`
4. Versión de PyTorch instalada
