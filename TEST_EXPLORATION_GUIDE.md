# 🎮 Guía Rápida: Cómo Cambiar el Modo de Evaluación en test.py

## ✅ RESPUESTA CORTA

**NO necesitas cambiar código directamente.** Ahora hay un **parámetro `--exploration`** que puedes usar desde línea de comandos.

---

## 🎯 Cómo Usar

### **Modo 1: EVALUACIÓN DETERMINISTA (default)**
```bash
python3.8 test.py --model model.mdl --episodes 10
```
✅ El Hopper toma la acción MEDIA (determinista)
✅ Sin exploración (posición congelada si el modelo así lo aprendió)
✅ Esto es lo que VES por defecto (comportamiento congelado)

---

### **Modo 2: EVALUACIÓN CON EXPLORACIÓN (nuevo)**
```bash
python3.8 test.py --model model.mdl --episodes 10 --exploration
```
✅ El Hopper EXPLORA (acciones aleatorias con varianza)
✅ Comportamiento menos determinista
✅ Ver si el modelo realmente aprendió patrones o solo memorizó

---

## 📋 Parámetros Completos

```bash
# Evaluación básica (determinista)
python3.8 test.py --model model.mdl

# Evaluación con exploración
python3.8 test.py --model model.mdl --exploration

# Evaluación con más episodios
python3.8 test.py --model model.mdl --episodes 100

# Evaluación con video
python3.8 test.py --model model.mdl --episodes 5 --record-video --record-every 1

# Evaluación con exploración + video + múltiples episodios
python3.8 test.py --model model.mdl --episodes 20 --exploration --record-video --record-every 2

# Training mode (dentro del test.py)
python3.8 test.py --train --episodes 5000

# Training con exploración (por defecto ya lo tiene)
python3.8 test.py --train --episodes 5000 --save-every 500 --device cuda
```

---

## 🔍 Cómo Ver la Diferencia

### **Paso 1: Ejecutar en modo determinista**
```bash
python3.8 test.py --model model.mdl --episodes 3
```

Verás en console:
```
======================================================================
MODO: EVALUACIÓN DETERMINISTA (deterministic - mean actions)
======================================================================

Episode: 1/3 | Return: 5.23
Episode: 2/3 | Return: 5.23
Episode: 3/3 | Return: 5.23

Average return over 3 episodes: 5.23
```

⚠️ Si el `Return` es **muy similar en todos los episodios** → Modelo tomó acciones fijas

---

### **Paso 2: Ejecutar en modo con exploración**
```bash
python3.8 test.py --model model.mdl --episodes 3 --exploration
```

Verás en console:
```
======================================================================
MODO: EVALUACIÓN CON EXPLORACIÓN (stochastic - actions aleatorias)
======================================================================

Episode: 1/3 | Return: 4.56
Episode: 2/3 | Return: 7.89
Episode: 3/3 | Return: 6.12

Average return over 3 episodes: 6.19
```

✅ Si el `Return` **varía en cada episodio** → Hay exploración

---

## 🧠 Interpretación de Resultados

| Escenario | Determinista | Con Exploración | Conclusión |
|-----------|---|---|---|
| Mismos Returns | ✅ 5.2, 5.2, 5.2 | ❌ 5.2, 5.2, 5.2 | Modelo tiene política fija |
| Returns variados | ⚠️ 5.2, 5.2, 5.2 | ✅ 4.1, 7.3, 6.5 | Modelo aprende pero con exploración mejora |
| Returns muy altos | ✅ 150, 150, 150 | ✅ 145, 155, 148 | Modelo entrenado bien |
| Returns muy bajos | ❌ 1.2, 1.2, 1.2 | ❌ 0.8, 1.5, 1.1 | Modelo NO entrenado bien |

---

## 📊 Qué Cambió en el Código

### **Antes:**
```python
# En test.py línea 153
action, _ = agent.get_action(state, evaluation=True)  # Siempre determinista
```

### **Después:**
```python
# En test.py línea ~155
evaluation_mode = not args.exploration
action, _ = agent.get_action(state, evaluation=evaluation_mode)

# Si --exploration: evaluation=False (estocástico)
# Si sin --exploration: evaluation=True (determinista)
```

---

## ⚡ Resumen de Comandos Útiles

```bash
# Ver ayuda de todos los parámetros
python3.8 test.py --help

# Test determinista (congelado si entrenamiento insuficiente)
python3.8 test.py --model model.mdl

# Test con exploración (ver variabilidad)
python3.8 test.py --model model.mdl --exploration

# Test con video para visualizar diferencia
python3.8 test.py --model model.mdl --episodes 3 --record-video
python3.8 test.py --model model.mdl --episodes 3 --record-video --exploration

# Comparar ambos modos
# Terminal 1:
python3.8 test.py --model model.mdl --episodes 5 --video-folder videos_determinista

# Terminal 2:
python3.8 test.py --model model.mdl --episodes 5 --exploration --video-folder videos_exploracion
```

---

## 💡 Recomendación

Para diagnosticar tu problema de "modelo congelado":

1. **Ejecuta sin exploración (default):**
   ```bash
   python3.8 test.py --model model.mdl --episodes 5
   ```
   Anota los valores de `Return`

2. **Ejecuta con exploración:**
   ```bash
   python3.8 test.py --model model.mdl --episodes 5 --exploration
   ```
   Anota los valores de `Return`

3. **Compara:**
   - Si ambos tienen `Return ~1-5` → Modelo no fue entrenado bien
   - Si sin exploración tiene `Return ~5` pero con exploración `~20` → Modelo aprende pero necesita más regularización
   - Si ambos tienen `Return ~200` → Modelo está bien entrenado

---

## 🎬 Ejemplo Completo

```bash
# Entrenamiento (en Colab con GPU)
python3.8 train.py --n-episodes 10000 --print-every 1000 --device cuda

# Guardar modelo
# (Se guarda automáticamente como model.mdl)

# Test determinista (ver si está congelado)
python3.8 test.py --model model.mdl --episodes 3

# Test con exploración (ver si hay variabilidad)
python3.8 test.py --model model.mdl --episodes 3 --exploration

# Test con video para análisis visual
python3.8 test.py --model model.mdl --episodes 2 --record-video --exploration
```

---

¡Listo! Ahora puedes cambiar el modo de evaluación sin modificar código, solo usando el parámetro `--exploration`. 🚀
