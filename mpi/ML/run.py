#!/usr/bin/env python3
"""Точка входа для запуска сравнения методов реконструкции MPI"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_pipeline


if __name__ == '__main__':
    # Параметры запуска
    NUM_SAMPLES = 5000        # Количество образцов для генерации
    TRAIN_MODELS = True       # False - загружать сохраненные модели, True - обучать новые

    results = run_pipeline(num_samples=NUM_SAMPLES, train_models=TRAIN_MODELS)

    if results:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("=" * 70)

        for result in results:
            distance = result['distance']
            print(f"\nРасстояние: {distance:.3f}")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}дБ, FWHM={m['fwhm']:.2f}px, время={m['time']:.4f}c")