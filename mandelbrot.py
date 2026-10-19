"""
Mandelbrot-Fraktal: CPU vs. GPU (CUDA) mit Numba.

Installation (Python 3.10 oder neuer):
    pip install numba-cuda[cu13] numpy matplotlib

Start:
    python mandelbrot_cuda.py
"""
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba import cuda

# Bildgröße und Ausschnitt
WIDTH, HEIGHT = 1920*5, 1080*5
MAX_ITER = 500
X_MIN, X_MAX = -2.2, 0.8
Y_MIN, Y_MAX = -0.85, 0.85


# ---------------------------------------------------------------- GPU-Kernel
@cuda.jit
def mandelbrot_gpu(image, x_min, x_max, y_min, y_max, max_iter):
    # Jeder GPU-Thread berechnet genau ein Pixel
    px, py = cuda.grid(2)
    h, w = image.shape
    if px >= w or py >= h:
        return

    cr = x_min + px * (x_max - x_min) / w
    ci = y_min + py * (y_max - y_min) / h
    zr = 0.0
    zi = 0.0
    n = 0
    while n < max_iter and zr * zr + zi * zi <= 4.0:
        zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
        n += 1

    # Weicher Farbverlauf statt harter Stufen
    if n < max_iter:
        mag = math.sqrt(zr * zr + zi * zi)
        image[py, px] = n + 1.0 - math.log(math.log(mag)) / math.log(2.0)
    else:
        image[py, px] = 0.0


# ---------------------------------------------------------------- CPU-Version
@njit
def mandelbrot_cpu(image, x_min, x_max, y_min, y_max, max_iter):
    h, w = image.shape
    for py in range(h):
        for px in range(w):
            cr = x_min + px * (x_max - x_min) / w
            ci = y_min + py * (y_max - y_min) / h
            zr = 0.0
            zi = 0.0
            n = 0
            while n < max_iter and zr * zr + zi * zi <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                n += 1
            if n < max_iter:
                mag = math.sqrt(zr * zr + zi * zi)
                image[py, px] = n + 1.0 - math.log(math.log(mag)) / math.log(2.0)
            else:
                image[py, px] = 0.0


def run_gpu():
    d_image = cuda.device_array((HEIGHT, WIDTH), dtype=np.float32)
    threads = (16, 16)
    blocks = (math.ceil(WIDTH / threads[0]), math.ceil(HEIGHT / threads[1]))

    # Erster Aufruf kompiliert den Kernel -> nicht mitmessen
    mandelbrot_gpu[blocks, threads](d_image, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    cuda.synchronize()

    t0 = time.perf_counter()
    mandelbrot_gpu[blocks, threads](d_image, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    cuda.synchronize()
    t = time.perf_counter() - t0
    return d_image.copy_to_host(), t


def run_cpu():
    image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mandelbrot_cpu(image[:2, :2], X_MIN, X_MAX, Y_MIN, Y_MAX, 10)  # kompilieren
    t0 = time.perf_counter()
    mandelbrot_cpu(image, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    return image, time.perf_counter() - t0


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("Keine CUDA-GPU gefunden (Treiber/numba-cuda pruefen).")
    try:
        name = cuda.get_current_device().name
        print(f"GPU: {name.decode() if isinstance(name, bytes) else name}")
    except AttributeError:
        pass
    print(f"Bild: {WIDTH} x {HEIGHT} Pixel, {MAX_ITER} Iterationen max.\n")

    img_gpu, t_gpu = run_gpu()
    print(f"GPU (CUDA):        {t_gpu * 1000:8.1f} ms")

    #img_cpu, t_cpu = run_cpu()
    #print(f"CPU (1 Kern, JIT): {t_cpu * 1000:8.1f} ms")
    #print(f"\nGPU ist {t_cpu / t_gpu:.1f}x schneller")

    #diff = np.max(np.abs(img_gpu - img_cpu))
    #print(f"Max. Abweichung GPU vs. CPU: {diff:.4f}")

    plt.figure(figsize=(12, 7))
    plt.imshow(img_gpu, cmap="twilight_shifted", extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    #plt.title(f"Mandelbrot auf der GPU: {t_gpu * 1000:.1f} ms  (CPU: {t_cpu * 1000:.0f} ms)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()