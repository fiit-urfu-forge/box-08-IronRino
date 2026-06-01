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
    # Параметры запуска
    # Размер синтетического обучающего датасета. SM-путь будет содержать
    # NUM_SAMPLES изображений (80% train + 20% test), physical-путь —
    # min(1500, NUM_SAMPLES // 2). Поднято 3000 → 5000: supervised-
    # модели на 200-3000 образцах недообучались (физический режим
    # давал SSIM 0.1-0.3 у CNN/MoDL/Diffusion). 5000 → 4000 train —
    # реальный объём, на котором U-Net учит широкое распределение
    # фантомов. Дешёво по compute (генерация ~3-5 минут на CPU).
    NUM_SAMPLES = 5000
    TRAIN_MODELS = True         # False — загружать сохранённые веса
    # Test-time оптимизация PMCNet (data-free, на одно измерение).
    # Поднято 3000 → 5000: smoke-test на 1500 итерациях дал SSIM 0.49
    # для Paper после исправления output_scale, а loss всё ещё падал
    # → больше итераций → выше SSIM. На синтетических данных, где
    # истинная физика совпадает с моделью, PMCNet может сходиться к
    # SSIM 0.7-0.9 при достаточном compute-budget'е.
    # На реальном CPU: ~7-12 минут × 5 PMCNet-вариантов × 11 фантомов
    # ≈ 5-10 часов общий PMCNet-проход. Уменьшите при тестировании.
    PMCNET_ITER = 5000

    # Регенерация синтетического датасета.
    # По умолчанию False: при первом запуске данные генерируются и
    # сохраняются в DATA/dataset/synthetic_*.npy; при повторных
    # запусках они загружаются с диска (секунды вместо минут).
    # Поставьте True, если меняли типы фантомов, размеры частиц или
    # хотите принудительно пересоздать с нуля.
    FORCE_REGENERATE_DATASET = False

    # Валидация на OpenMPIData (https://github.com/MagneticParticleImaging/OpenMPIData.jl)
    # ВЫКЛЮЧЕНА по умолчанию. Чтобы включить — поставьте True. Данные
    # должны лежать в `mpi/ChineseData/OpenMPIData/` со структурой:
    #     calibrations/   ← *.mdf (системные функции)
    #     measurements/   ← подпапки backgroundDrift/, concentrationPhantom/,
    #                        resolutionPhantom/, rotationPhantom/, shapePhantom/
    #                        внутри каждой *.mdf
    VALIDATE_OPENMPI = False

    results = run_pipeline(
        num_samples=NUM_SAMPLES,
        train_models=TRAIN_MODELS,
        pmcnet_iterations=PMCNET_ITER,
        validate_openmpi=VALIDATE_OPENMPI,
        force_regenerate_dataset=FORCE_REGENERATE_DATASET,
    )

    synth = results.get('synthetic') if isinstance(results, dict) else results
    if synth:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ (батарея фантомов × методов):")
        print("=" * 70)
        for result in synth:
            label = result.get('label', 'unnamed')
            print(f"\n[{label}]")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name:<22} SSIM={m['ssim']:.4f}  "
                      f"PSNR={m['psnr']:.2f}дБ  FWHM={m['fwhm']:.2f}px  "
                      f"time={m['time']:.4f}с")
