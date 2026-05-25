#!/usr/bin/env python3
"""Точка входа: запускает полный pipeline сравнения методов MPI.

Реальная логика — в `src.pipeline.run_pipeline`. Здесь только параметры
запуска и красивая распечатка сводных результатов.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import run_pipeline  # noqa: E402


if __name__ == '__main__':
    # Параметры запуска (минимальные — для отладки)
    NUM_SAMPLES = 200       # размер синтетического обучающего датасета
    TRAIN_MODELS = True     # False — загружать сохранённые веса
    PMCNET_ITER = 50        # итераций оптимизации на одно измерение

    results = run_pipeline(
        num_samples=NUM_SAMPLES,
        train_models=TRAIN_MODELS,
        pmcnet_iterations=PMCNET_ITER,
    )

    synth = results.get('synthetic') if isinstance(results, dict) else results
    if synth:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ (синтетика, двух-капельные фантомы):")
        print("=" * 70)
        for result in synth:
            distance = result['distance']
            print(f"\nРасстояние: {distance:.3f}")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name:<22} SSIM={m['ssim']:.4f}  "
                      f"PSNR={m['psnr']:.2f}дБ  FWHM={m['fwhm']:.2f}px  "
                      f"time={m['time']:.4f}с")
