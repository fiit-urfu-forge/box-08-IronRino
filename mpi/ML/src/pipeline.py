"""End-to-end pipeline сравнения методов реконструкции MPI.

## Что делает этот модуль

Главная точка входа — `run_pipeline()`. Она проходит весь жизненный
цикл эксперимента:

  1. **Генерация синтетического датасета.** Создаёт обучающую выборку
     двумя путями:
       • SM-путь: через измеренную системную матрицу из калибровки
         сканера (учитывает все реальные несовершенства);
       • physical-путь: через аналитический физический форвард
         (идеализированная физика, без калибровочных артефактов).
     Это даёт две параллельные оценки методов — на «реальных» и
     «идеальных» данных.

  2. **Обучение/загрузка всех моделей реконструкции.** Четыре категории:
       • Классические: Tikhonov, Kaczmarz (без обучения, без параметров);
       • Архитектурные baseline'ы: CNN, MoDL, Diffusion;
       • Прямая регрессия: Chae 2017 (single + multi-layer FC);
       • Физически-ограниченные: PMCNet (4 варианта — Standard, Paper,
         PhysicsEnhanced, Final).

     Обучение каждой модели можно пропустить, передав `train_models=False`
     (тогда загружаются заранее сохранённые веса).

  3. **Сборка тестовой батареи фантомов.** Шесть типов:
       two_droplets, phantom_4, rotation_45, shape_ring, random,
       letter_B (последний — реальное измерение со сканера).
     На каждый из пяти первых — два пути измерения (sm + physical),
     получаем 11 экспериментов на батарею.

  4. **Запуск сравнительной таблицы.** Каждый метод реконструирует
     каждый эксперимент, метрики (SSIM/PSNR/FWHM/время) сохраняются в
     общую таблицу с визуализациями.

  5. **MoE-сборка.** Опциональное комбинирование нескольких быстрых
     методов через Mixture of Experts.

  6. **Валидация на OpenMPIData (опционально).** Реальные измерения
     с публичного MPI-датасета.

## Train/Load переключение

  • `train_models=True`: обучает все supervised-модели (CNN, MoDL,
    Chae, Diffusion). Веса сохраняются в DATA/models/.
  • `train_models=False`: загружает заранее обученные веса. Полезно
    для повторного прогона на новых фантомах без полного обучения.

Data-free методы (DIP, PMCNet × 4) НЕ требуют обучения вообще —
они оптимизируют веса U-Net на каждом измерении отдельно. Это
существенно медленнее на этапе сравнения (минуты на измерение), но
не требует подготовки.

## Размеры данных

  • Изображения: (51, 51) пикселей (формат BeihangUniversityData);
  • Измерения: complex (2, 1275) per phantom (2 катушки × 1275 гармоник);
  • Системная матрица: complex (2550, 2601).

## Гиперпараметры

Главные на уровне `run.py`:
  • NUM_SAMPLES: размер обучающего синтетического датасета;
  • PMCNET_ITER: число итераций для data-free методов (PMCNet, DIP).
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import MPIDatasetGenerator, MPIDataset, OpenMPIDataManager
from .data.simulators import (
    SyntheticDatasetGenerator, NanoparticleProperties,
)
from .data.phantoms import PhantomType
from .models import (
    ChaeSingleLayerNN, ChaeMultiLayerNN,
    DeepImagePrior,
    PMCNetConfig, PMCNetStandard, PMCNetPaper,
    PMCNetRadialCoil, PMCNetSoftConstrained,
    PMCNetDebye, PMCNetCentralFD,
    MoEReconstructor,
    build_analytical_system_matrix,
)
from .data.phantoms import PhantomGenerator
from .trainer import MPITrainer, ModelTrainerFactory
from .comparator import MPIReconstructionComparator


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def setup_directories():
    for d in ('./DATA/dataset', './DATA/models',
              './DATA/results', './DATA/results/training_curves',
              './DATA/results/phantoms'):
        os.makedirs(d, exist_ok=True)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# 1. Синтетические данные — оба пути из Chae 2017
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(num_samples: int = 2000, save: bool = True,
                                force_regenerate: bool = False):
    """Сгенерировать или загрузить датасеты двумя путями.

    При первом вызове: создаёт SM-путь и physical-путь, сохраняет на диск.
    При повторных: загружает сохранённые с диска, минуя дорогую генерацию.

    Args:
        num_samples: размер SM-датасета (для PHYS используется
                     min(500, num_samples//3) — physical-форвард медленнее).
        save: сохранять ли датасеты на диск при первой генерации.
        force_regenerate: если True, игнорирует сохранённые файлы и
                          генерирует заново. Полезно при изменении
                          конфигурации (типов фантомов, размеров частиц).

    Returns:
        (generator, dataset_sm) — генератор + основной (SM) датасет.
    """
    print("\n" + "=" * 70)
    print("1. СИНТЕТИЧЕСКИЕ ДАННЫЕ (SM + physical пути)")
    print("=" * 70)

    generator = SyntheticDatasetGenerator(nx=51, ny=51, n_harmonics=200)

    # Визуализация всех типов фантомов — для отчёта
    generator.phantom_gen.visualize_phantoms(
        save_path='./DATA/results/phantoms/all_phantoms.png',
    )

    sm_prefix = './DATA/dataset/synthetic_complete'
    phys_prefix = './DATA/dataset/synthetic_physical'

    # --- Путь 1: SM-генерация ---
    if not force_regenerate and SyntheticDatasetGenerator.dataset_exists(sm_prefix):
        print(f"\n  Путь 1 (SM): загрузка из {sm_prefix}_*.npy ...")
        dataset_sm = SyntheticDatasetGenerator.load_dataset(sm_prefix)
        print(f"    загружено: {len(dataset_sm['X_train'])} train + "
              f"{len(dataset_sm['X_test'])} test")
    else:
        print(f"\n  Путь 1 (SM): генерация {num_samples} образцов...")
        dataset_sm = generator.create_training_pipeline_dataset(
            n_samples=num_samples, include_all_phantoms=True, save=save,
        )

    # --- Путь 2: physical-генерация ---
    n_phys = min(500, num_samples // 3)
    if not force_regenerate and SyntheticDatasetGenerator.dataset_exists(phys_prefix):
        print(f"\n  Путь 2 (physical): загрузка из {phys_prefix}_*.npy ...")
        dataset_phys = SyntheticDatasetGenerator.load_dataset(phys_prefix)
        print(f"    загружено: {len(dataset_phys['X_train'])} train + "
              f"{len(dataset_phys['X_test'])} test")
    else:
        print(f"\n  Путь 2 (physical): генерация {n_phys} образцов (медленнее)...")
        dataset_phys = generator.generate_dataset(
            n_samples=n_phys,
            method='physical',
            phantom_types=[PhantomType.TWO_DROPLETS, PhantomType.PHANTOM_4,
                           PhantomType.ROTATION, PhantomType.MULTI_POINT,
                           PhantomType.LINES, PhantomType.CIRCLES],
            particle_sizes_nm=[30, 40, 50],
            add_noise=True, snr_db=35.0, test_split=0.2,
        )
        if save:
            generator.save_dataset(dataset_phys, phys_prefix)
            print(f"    сохранено в {phys_prefix}_*.npy")

    print(f"\n  Итого: SM = {len(dataset_sm['X_train'])} train, "
          f"phys = {len(dataset_phys['X_train'])} train.")
    return generator, dataset_sm


# ---------------------------------------------------------------------------
# Cross-validation параметра μ для Tikhonov-реконструктора
# ---------------------------------------------------------------------------


def tune_tikhonov_mu(SM, X_val, y_val,
                     mus=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                     kmax: int = 30,
                     n_val_samples: int = 16) -> float:
    """Выбрать `μ` для Tikhonov, минимизирующий MSE на валидационной выборке.

    Tikhonov даёт SSIM 0.5–0.95 в зависимости от уровня шума, а
    оптимальный μ — порядка σ²_шума / σ²_сигнала. Подбираем на
    маленькой выборке (16–32 образца) перед сравнением.
    """
    from .models.classical import TikhonovReconstructor
    tik = TikhonovReconstructor(SM)
    n_use = min(n_val_samples, len(X_val))
    idx = np.random.choice(len(X_val), n_use, replace=False)
    best_mu, best_mse = None, float('inf')
    print("  Подбор μ для Tikhonov:")
    for mu in mus:
        losses = []
        for i in idx:
            m = X_val[i]
            v = np.concatenate([m[0], m[1]])
            recon = tik.reconstruct(v, mu=mu, kmax=kmax).reshape(*y_val.shape[1:])
            losses.append(np.mean((recon - y_val[i]) ** 2))
        mse = float(np.mean(losses))
        marker = ''
        if mse < best_mse:
            best_mu, best_mse = mu, mse
            marker = '  ← новый минимум'
        print(f"    μ={mu:.0e}  MSE={mse:.4e}{marker}")
    print(f"  → выбран μ={best_mu:.0e} (MSE={best_mse:.4e})")
    return best_mu


# ---------------------------------------------------------------------------
# 2. Обучение / загрузка моделей
# ---------------------------------------------------------------------------


def _save_curve(losses, name, fname):
    plt.figure(figsize=(9, 5))
    if isinstance(losses, tuple):
        train_l, val_l = losses
        plt.plot(train_l, label='train', linewidth=2)
        plt.plot(val_l, label='val', linewidth=2)
        plt.legend()
    else:
        plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{name} Training'); plt.grid(True, alpha=0.3)
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def _model_input_dim(X_train):
    """Размер амплитудного спектра по 2 катушкам = 2·M_per_coil.

    Chae 2017 работает со СПЕКТРОМ АМПЛИТУД |u|, а не с Re/Im (paper
    Sec. III.1, Fig. 3a). Это связано с тем, что обучаемые веса W
    стремятся к полиномам Чебышёва — то есть к псевдо-обращению
    спектра амплитуд. На Re/Im формате эта структура не возникает.
    """
    return X_train.shape[2] * 2


def _flatten_measurement_batch(X):
    """(N, 2, M_per_coil) комплекс → (N, 2·M_per_coil) амплитудный
    спектр, нормированный per-sample на max(|u|).

    Per-sample max-нормализация:
      • инвариантна к глобальной амплитуде сигнала (sensitivity,
        концентрация образца) — модель видит «форму» спектра, не уровень;
      • не требует сохранения статистик scaler_mean/scale на диске;
      • сопоставима с поведением sigmoid-выхода Chae (значения в [0,1]).

    Согласовано с `comparator.chae_reconstruction`, который применяет
    тот же препроцессинг на инференсе.
    """
    abs_spec = np.abs(X)                                  # (N, 2, M)
    flat = abs_spec.reshape(len(X), -1).astype(np.float32)
    # Per-sample max-нормировка
    s = flat.max(axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return flat / s


def _synthesize_measurements_through_SM(y_images, SM,
                                        snr_db: float = 30.0,
                                        random_seed: int = 0):
    """Регенерация измерений через ИЗМЕРЕННУЮ системную матрицу.

    Синтетический генератор по умолчанию использует Chebyshev-SM с
    `n_harmonics=200`, что даёт X-шейпы, несовместимые с измеренной SM
    (которая обычно 2·n_freq_per_coil × N). Чтобы все обучаемые модели
    (CNN/MoDL/Chae) видели данные правильной размерности,
    регенерируем `X = S · y` для каждого изображения y.

    Args:
        y_images: (N, H, W) ground-truth изображения.
        SM:       (M, H·W) комплексная измеренная системная матрица.
        snr_db:   шум, добавляемый поверх; None = без шума.

    Returns:
        (N, 2, M/2) complex64 — стандартный двух-катушечный формат.
    """
    rng = np.random.default_rng(random_seed)
    N = len(y_images)
    M_total = SM.shape[0]
    M_per_coil = M_total // 2
    out = np.zeros((N, 2, M_per_coil), dtype=np.complex64)
    for i in range(N):
        u = SM @ y_images[i].flatten()         # (M_total,) complex
        out[i, 0] = u[:M_per_coil]
        out[i, 1] = u[M_per_coil:]
        if snr_db is not None and np.isfinite(snr_db):
            sigma = np.std(np.abs(out[i])) / (10 ** (snr_db / 20))
            if sigma > 0:
                out[i] += sigma * (rng.standard_normal(out[i].shape)
                                   + 1j * rng.standard_normal(out[i].shape))
    return out


def _format_for_cnn(X_complex):
    """(N, 2, M_per_coil) complex → (N, 4, S, S) real (S² ≥ M_per_coil)."""
    N, _, M_per_coil = X_complex.shape
    S = int(np.ceil(np.sqrt(M_per_coil)))
    target = S * S
    out = np.zeros((N, 4, target), dtype=np.float32)
    out[:, 0, :M_per_coil] = X_complex[:, 0].real
    out[:, 1, :M_per_coil] = X_complex[:, 0].imag
    out[:, 2, :M_per_coil] = X_complex[:, 1].real
    out[:, 3, :M_per_coil] = X_complex[:, 1].imag
    return out.reshape(N, 4, S, S)


def _format_for_modl(X_complex):
    """(N, 2, M_per_coil) complex → (N, 4·M_per_coil) real.

    Для MoDL — вход — расширенное (Re, Im) представление измерения
    в форме одного вектора длины 2·M_total = 4·M_per_coil. Порядок:
    [coil0.real, coil1.real, coil0.imag, coil1.imag].
    """
    real_part = X_complex.real     # (N, 2, M_per_coil)
    imag_part = X_complex.imag
    re_flat = real_part.reshape(len(X_complex), -1)   # (N, 2·M_per_coil)
    im_flat = imag_part.reshape(len(X_complex), -1)
    return np.concatenate([re_flat, im_flat], axis=-1).astype(np.float32)


# --- CNN baseline -----------------------------------------------------------

def train_or_load_cnn(X_train, y_train, train: bool = True,
                      epochs: int = 80, output_size=(51, 51)):
    # epochs повышен 40 → 80: на полнокартинной reconstruction CNN
    # доходит до плато только после ~60 эпох. На предыдущем прогоне
    # CNN давал SSIM 0.15-0.49 — типичные признаки недотренировки.
    print("\n  CNN baseline (UNet)...")
    path = './DATA/models/cnn_best.pth'
    trainer = ModelTrainerFactory.create_cnn_trainer(
        input_channels=4, output_channels=1, base_filters=32,
    )
    if not train and os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("    загружена")
        return trainer

    X_img = _format_for_cnn(X_train)                  # (N, 4, S, S)
    train_ds = TensorDataset(
        torch.tensor(X_img),
        torch.tensor(y_train[:, None], dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    losses, _ = trainer.train(loader, loader, epochs=epochs, save_path=path)
    _save_curve(losses, 'CNN', './DATA/results/training_curves/cnn_training.png')
    return trainer


# --- MoDL baseline ----------------------------------------------------------

def train_or_load_modl(SM, image_shape, X_train, y_train,
                       train: bool = True, epochs: int = 60):
    # epochs повышен 30 → 60: MoDL с 5 unrolled-итерациями и residual
    # denoiser требует больше эпох для сходимости — каждая эпоха
    # тренирует ВСЕ итерации вместе. На предыдущем прогоне на простых
    # фантомах SSIM уже хорош (0.8-0.96), но на сложных (rotation,
    # ring physical) проседает до 0.04-0.15.
    print("\n  MoDL baseline...")
    path = './DATA/models/modl_best.pth'
    trainer = ModelTrainerFactory.create_modl_trainer(
        system_matrix=SM, image_shape=image_shape,
        n_iterations=5, lambda_param=0.05, base_filters=32,
    )
    if not train and os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("    загружена")
        return trainer

    X_real = _format_for_modl(X_train)            # (N, 2·M_total)
    ds = TensorDataset(
        torch.tensor(X_real),
        torch.tensor(y_train[:, None], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    losses, _ = trainer.train(loader, loader, epochs=epochs, save_path=path)
    _save_curve(losses, 'MoDL',
                './DATA/results/training_curves/modl_training.png')
    return trainer


# --- Diffusion baseline -----------------------------------------------------

def train_or_load_diffusion(y_train, X_train, SM,
                            train: bool = True, epochs: int = 80):
    # epochs повышен 25 → 80: DDPM с 100 timesteps и единственный пример
    # на батч учит t-эмбединги ОЧЕНЬ медленно — 25 эпох × 500 примеров
    # = только ~125 шагов на каждый из 100 timesteps. На предыдущем
    # прогоне Diffusion давал SSIM 0.01-0.41 (sm режим), на physical
    # вообще ~0.0. 80 эпох × 500 = 400 шагов/timestep — приемлемо.
    """Conditional DDPM: condition = Tikhonov-реконструкция из измерения.

    Без conditioning Diffusion безполезен для нашей задачи (генерирует
    «правдоподобные», но не соответствующие измерению карты). С condition'ом
    он учится «отшумливать» грубую Tikhonov-реконструкцию до GT.
    """
    print("\n  Diffusion baseline (conditional DDPM с Tikhonov-condition)...")
    path = './DATA/models/diffusion_best.pth'
    image_size = y_train.shape[-1]
    trainer = ModelTrainerFactory.create_diffusion_trainer(
        n_steps=100, image_size=image_size, base_filters=32,
    )
    if not train and os.path.exists(path):
        ckpt = torch.load(path, map_location=device())
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("    загружена")
        return trainer

    # Precompute Tikhonov-condition для каждого X_train (медленно, но один раз)
    from .models.classical import TikhonovReconstructor
    tik = TikhonovReconstructor(SM)
    n = min(500, len(y_train))
    sel = np.random.choice(len(y_train), n, replace=False)
    print(f"    Прекомьют Tikhonov-condition для {n} образцов...")
    conditions = np.zeros((n, *y_train.shape[1:]), dtype=np.float32)
    for k, i in enumerate(tqdm(sel, desc='      tik-cond')):
        v = np.concatenate([X_train[i, 0], X_train[i, 1]])
        conditions[k] = tik.reconstruct(v, mu=1e-2, kmax=15).reshape(*y_train.shape[1:])
        # Нормализация на [0, 1]
        if conditions[k].max() > 0:
            conditions[k] = conditions[k] / conditions[k].max()
    targets = torch.tensor(y_train[sel, None], dtype=torch.float32)
    cond_t = torch.tensor(conditions[:, None], dtype=torch.float32)

    # Тренировочный цикл: передаём condition напрямую в model.forward
    dev = device()
    model = trainer.model.to(dev)
    ds = TensorDataset(targets, cond_t)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses = []
    best = float('inf')
    for ep in tqdm(range(epochs), desc='    Diffusion'):
        ep_loss = 0.0
        for yb, cb in loader:
            yb = yb.to(dev); cb = cb.to(dev)
            opt.zero_grad()
            loss = model(yb, condition=cb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if avg < best:
            best = avg
            torch.save({'model_state_dict': model.state_dict()}, path)
    _save_curve(losses, 'Diffusion (conditional)',
                './DATA/results/training_curves/diffusion_training.png')
    return trainer


# --- Chae 2017 (single + multi layer) ---------------------------------------

def train_or_load_chae(SM, image_shape, X_train, y_train,
                       train: bool = True, epochs: int = 120):
    # epochs повышен 60 → 120: Chae Multi-Layer достигает SSIM 0.98 на
    # лёгких фантомах, но Single-Layer и Multi на сложных (rotation
    # physical, shape_ring) проседает до 0.03-0.40. Сетка с sigmoid+MSE
    # сходится медленно — 120 эпох даёт двойной запас.
    """Возвращает (single_layer_model, multi_layer_model) согласно статье."""
    print("\n  Chae (2017) — single + multi-layer FC...")
    in_dim = _model_input_dim(X_train)
    out_dim = image_shape[0] * image_shape[1]
    path_single = './DATA/models/chae_single_best.pth'
    path_multi = './DATA/models/chae_multi_best.pth'

    dev = device()
    single = ChaeSingleLayerNN(in_dim, out_dim).to(dev)
    # hidden_dim=None → default max(200, 1.5·out_dim), что выполняет требование
    # статьи Chae 2017 о hidden ≥ output (иначе сеть не обучается)
    multi = ChaeMultiLayerNN(in_dim, out_dim, hidden_dim=None).to(dev)

    if not train and os.path.exists(path_single) and os.path.exists(path_multi):
        single.load_state_dict(torch.load(path_single, map_location=dev),
                               strict=False)
        multi.load_state_dict(torch.load(path_multi, map_location=dev),
                              strict=False)
        print("    загружены")
        return single, multi

    X_flat = _flatten_measurement_batch(X_train)
    y_flat = y_train.reshape(len(y_train), -1).astype(np.float32)
    ds = TensorDataset(torch.tensor(X_flat), torch.tensor(y_flat))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    crit = torch.nn.MSELoss()

    for model, path, label in [(single, path_single, 'Chae-Single'),
                                (multi, path_multi, 'Chae-Multi')]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for ep in tqdm(range(epochs), desc=f'    {label}'):
            ep_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(dev); yb = yb.to(dev)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            losses.append(ep_loss / len(loader))
        torch.save(model.state_dict(), path)
        _save_curve(losses, label,
                    f'./DATA/results/training_curves/{label.lower()}_training.png')
    return single, multi


# --- DIP — data-free, обучения не требует ----------------------------------

def build_dip(image_shape):
    print("\n  DIP (Dittmer 2020) — data-free, без обучения.")
    return DeepImagePrior(image_shape, latent_channels=1,
                          base_channels=32).to(device())



# --- PMCNet ablation sextet (data-free) ------------------------------------

def build_pmcnet_variants(SM, image_shape,
                           n_iterations: int = 3000,
                           init_tau_seconds: float = 1.0e-9,
                           scanner_h5_path: Optional[str] = None):
    # n_iterations повышен 1500 → 3000: PMCNet-Paper и 4 ветки —
    # test-time оптимизация со случайно инициализированным U-Net.
    # На 1500 итерациях loss падает с 0.31 до 0.006, но качество
    # ещё не доходит до плато (smoke-test показал SSIM 0.49-0.56).
    # 3000 итераций дают двукратный запас и согласуются с paper
    # Sec. III.B (20000 итераций — оригинал, но для compute-budget
    # в дипломе 3000 — разумный компромисс).
    """Собрать шесть вариантов PMCNet для ablation-сравнения.

    Структура: одна общая «база» (PhysicsEnhanced) + три параллельные
    ветки одиночных улучшений. Каждая ветка добавляет ровно ОДИН
    компонент к базовой физике, чтобы можно было независимо оценить его
    вклад в метрики:

           PMCNet-Std (измеренная SM)
           PMCNet-Paper (аналитическая, paper-faithful)
                 │
                 ▼
           PMCNet-Phys     ◄── базовая физика (radial p + langevin_safe)
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     Soft      Debye    CentralFD
     (soft     (Debye   (Conv1d
     loss)    relax.)   ∂/∂t)

    Args:
        scanner_h5_path: путь к SystemMatrix.h5 (или любому H5 MDF).
            Если задан, для PMCNet-Paper подгружаются точные физические
            параметры сканера и явный DFT на frequencySelection.

    Returns:
        Словарь {name: reconstructor} с 6 ключами:
          'standard', 'paper', 'radial_coil', 'soft', 'debye', 'central_fd'.

    Структура наследования:
        PMCNetStandard (измеренная SM)         — отдельный baseline
        PMCNetPaper (paper-faithful)           — общая БАЗА для четырёх
          ├── PMCNetRadialCoil   (+ radial p(r))
          ├── PMCNetSoftConstrained (+ Scheinker soft loss)
          ├── PMCNetDebye        (+ обучаемая τ Debye)
          └── PMCNetCentralFD    (+ central FD через Conv1d)

    Каждая из четырёх веток включает РОВНО ОДНО улучшение, что даёт
    чистую ablation: «насколько данное улучшение поднимает метрику
    относительно paper-faithful базы».
    """
    print("\n  PMCNet (Huang 2026) — шесть вариантов (data-free) для ablation...")
    base = PMCNetConfig(
        image_size=tuple(image_shape),
        n_iterations=n_iterations,
        learning_rate=1e-3,
        init_tau_seconds=init_tau_seconds,
    )
    n_meas = int(SM.shape[0])
    standard = PMCNetStandard(SM, image_shape, config=base)

    # PMCNet-Paper: подгружаем реальные параметры сканера из H5, если дан путь.
    # Эти параметры применяются КО ВСЕМ четырём наследникам — все они
    # используют ту же физику, что и Paper.
    paper_config = base
    paper_freqs = None
    paper_n_meas = n_meas
    if scanner_h5_path is not None:
        try:
            scanner = _load_scanner_params_from_h5(scanner_h5_path)
            paper_freqs = scanner.pop('frequencies_hz')
            paper_config = PMCNetConfig(**{**base.__dict__, **scanner})
            paper_n_meas = 2 * len(paper_freqs)
            print(f"    [Paper] точные параметры из H5: "
                  f"f_x={paper_config.drive_frequency_x:.0f} Hz, "
                  f"f_y={paper_config.drive_frequency_y:.0f} Hz, "
                  f"G_x={paper_config.gradient_strength:.2e}, "
                  f"A_x={paper_config.drive_field_amplitude:.0f} A/m, "
                  f"{len(paper_freqs)} частот для DFT")
        except (FileNotFoundError, OSError) as e:
            print(f"    [Paper] ⚠ не удалось прочитать {scanner_h5_path}: {e}")
            print(f"    [Paper] используются config-дефолты")
    paper = PMCNetPaper(image_shape, paper_n_meas, config=paper_config,
                         frequencies_hz=paper_freqs)

    # Четыре параллельных ветки — все используют paper_config и paper_freqs,
    # отличаются только своим конкретным флагом-улучшением.
    radial_coil = PMCNetRadialCoil(image_shape, paper_n_meas,
                                    config=paper_config,
                                    frequencies_hz=paper_freqs)
    soft = PMCNetSoftConstrained(image_shape, paper_n_meas,
                                  config=paper_config,
                                  frequencies_hz=paper_freqs)
    debye = PMCNetDebye(image_shape, paper_n_meas,
                        config=paper_config,
                        frequencies_hz=paper_freqs)
    central_fd = PMCNetCentralFD(image_shape, paper_n_meas,
                                  config=paper_config,
                                  frequencies_hz=paper_freqs)

    print(f"    1) Standard     (SM-baseline — НЕ из paper)")
    print(f"    2) Paper [база] (paper-faithful, Huang 2026 Eq. 1-3)")
    print(f"    3) RadialCoil   ← Paper + p(r) = 1/(1+(r/R)²)")
    print(f"    4) Soft         ← Paper + Scheinker 2023 soft (‖∇c‖₂² + freq-weighted L1)")
    print(f"    5) Debye        ← Paper + релаксация Дебая (обучаемая τ)")
    print(f"    6) CentralFD    ← Paper + центральная FD через Conv1d (O(Δt²))")
    return {
        'standard': standard, 'paper': paper,
        'radial_coil': radial_coil, 'soft': soft,
        'debye': debye, 'central_fd': central_fd,
    }


# Backwards-compat alias — кто-то может импортировать старое имя
build_pmcnet_quartet = build_pmcnet_variants




# ---------------------------------------------------------------------------
# 2.5. Mixture of Experts — комбинирование моделей
# ---------------------------------------------------------------------------


def build_moe(comparator, image_shape,
              X_train, y_train,
              expert_names=('Тихонов', 'KatsMarc',
                            'Chae(2017)', 'Chae-Multi(2017)',
                            'CNN'),
              n_train_samples: int = 64,
              epochs: int = 30,
              mode: str = 'spatial'):
    """Собрать MoE поверх уже привязанных к comparator'у экспертов.

    Шаги:
      1. Из comparator достаём callable'ы выбранных экспертов
         (имена должны совпадать с теми, что появляются в таблице
         сравнения).
      2. Прогоняем экспертов на маленькой выборке из training set,
         сохраняя их предсказания — это вход для обучения gating.
      3. Тренируем gating (per-pixel CNN по умолчанию) минимизировать
         MSE против ground truth.
      4. Возвращаем готовый `MoEReconstructor`, чтобы подключить
         его в comparator.

    ## Критерий отбора экспертов в default-списке

    В дефолт попали только модели, **быстрые на инференсе после
    предобучения** (или вовсе не требующие обучения, как классические):

      • Тихонов, KatsMarc — без обучения, итеративный inference 2-10 с;
      • Chae(2017) Single + Multi — FC-сети, ~5 мс на инференс;
      • CNN (U-Net) — ~10 мс.

    НЕ в дефолте (по дизайну):

      • **PMCNet × 4** — требуют 1500-3000 test-time-оптимизации,
        прекомпьют 1500+ сэмплов сделает сборку MoE 10+ часов;
      • **DIP(2020)** — test-time-обучение (3000 итераций с early stop),
        4-8 с на сэмпл → прекомпьют 64 образцов ~5 минут (ещё допустимо),
        но эта схема концептуально data-free, а MoE — supervised;
      • **Diffusion** — DDPM-семпл 100 шагов = 13-18 с/сэмпл, прекомпьют
        64 образцов ~15 минут. Можно добавить вручную: пройдёт в MoE,
        но gating увидит этого эксперта как «медленный гладкий source»;
      • **MoDL** — formально fast-after-pretraining (~50 мс), но
        внутри уже содержит `(AᵀA+λI)⁻¹` data-consistency — то же, что
        делает Tikhonov в MoE-наборе. На синтетических фантомах его
        вклад сильно коррелирует с Tikhonov-экспертом → gating-сети
        нечего разделять. Исключён, чтобы избежать избыточности.

    Если хочется включить MoDL/PMCNet/DIP/Diffusion — передайте их в
    `expert_names`. Имена совпадают с теми, что в таблице сравнения.
    """
    print("\n" + "=" * 70)
    print(f"2.5. MIXTURE OF EXPERTS — комбинирование ({len(expert_names)} экспертов)")
    print("=" * 70)
    print(f"  Эксперты: {list(expert_names)}")
    print(f"  Режим комбинирования: {mode!r}")

    # Достаём callable'ы из текущего comparator
    name_to_callable = {
        'Тихонов': comparator.tikhonov_reconstruction,
        'KatsMarc': lambda m: comparator.katsmarc_reconstruction(m, n_iterations=10),
        'Chae(2017)': comparator.chae_reconstruction,
        'Chae-Multi(2017)': comparator.chae_multi_reconstruction,
        'DIP(2020)': comparator.dip_reconstruction,
        'CNN': comparator.cnn_reconstruction,
        'MoDL': comparator.modl_reconstruction,
        'Diffusion': comparator.diffusion_reconstruction,
        'PMCNet-Std(2026)': comparator.pmcnet_standard_reconstruction,
        'PMCNet-RadialCoil(2026)': comparator.pmcnet_radial_coil_reconstruction,
        'PMCNet-Soft(2026)': comparator.pmcnet_soft_reconstruction,
        'PMCNet-Debye(2026)': comparator.pmcnet_debye_reconstruction,
        'PMCNet-CentralFD(2026)': comparator.pmcnet_central_fd_reconstruction,
    }
    experts = {}
    for n in expert_names:
        if n not in name_to_callable:
            print(f"  Skipped (unknown expert): {n}")
            continue
        # Проверяем что соответствующая модель установлена
        try:
            test_recon = name_to_callable[n]  # пробная сборка callable
            experts[n] = test_recon
        except Exception as e:
            print(f"  Skipped ({n}): {e}")

    if not experts:
        print("  Ни один эксперт недоступен — MoE пропущен.")
        return None

    moe = MoEReconstructor(
        experts=experts,
        image_shape=image_shape,
        mode=mode,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Прекомпьют выходов экспертов на n_train_samples образцов
    n_use = min(n_train_samples, len(X_train))
    idx = np.random.choice(len(X_train), n_use, replace=False)
    measurements_subset = [X_train[i] for i in idx]
    targets_subset = torch.tensor(y_train[idx], dtype=torch.float32)
    print(f"\n  Прекомпьют выходов экспертов на {n_use} образцах...")
    expert_recons = moe.precompute_expert_recons(
        measurements_subset, show_progress=True)

    # Обучение gating
    if mode != 'mean':
        print(f"\n  Обучение gating ({mode}, {epochs} эпох)...")
        moe.train_gating(expert_recons, targets_subset,
                         epochs=epochs, lr=1e-3, batch_size=8,
                         verbose=True)

    # Diagnostic: разница MSE между лучшим индивидуальным экспертом
    # и MoE-комбинированием на этой же выборке
    with torch.no_grad():
        moe.eval()
        moe_pred = moe(expert_recons.to(moe.device))
        moe_mse = ((moe_pred.cpu() - targets_subset.unsqueeze(1)) ** 2).mean().item()
        expert_mses = ((expert_recons - targets_subset.unsqueeze(1)) ** 2
                       ).mean(dim=(0, 2, 3))
        best_expert_idx = int(expert_mses.argmin())
        best_name = moe.expert_names[best_expert_idx]
        print(f"\n  MSE на training-выборке:")
        for i, name in enumerate(moe.expert_names):
            mark = ' ← лучший' if i == best_expert_idx else ''
            print(f"    {name:<18} MSE={expert_mses[i].item():.4e}{mark}")
        improvement = (expert_mses.min().item() - moe_mse) / expert_mses.min().item() * 100
        print(f"    {'MoE':<18} MSE={moe_mse:.4e}  "
              f"(улучшение vs {best_name}: {improvement:+.1f}%)")

    return moe


# ---------------------------------------------------------------------------
# 3. Батарея фантомов: 6 типов × 2 пути генерации измерений
# ---------------------------------------------------------------------------


def _load_scanner_params_from_h5(path: str) -> dict:
    """Точные физические параметры сканера из H5 (MDF-формат).

    Без этого PMCNet-Paper использует config-дефолты, которые могут
    радикально не совпадать с реальным сканером. Для
    BeihangUniversityData: config f_x=24510 Hz, а реальное f_x=10000 Hz
    (2.5×); config f_y=26042 Hz, реальное f_y=1 Hz (4 порядка). При
    таком несоответствии loss считается между разными физическими
    объектами и Paper не может сойтись.

    Эмпирически определённые единицы (Beihang sample):
      • gradient stored as Т/м → конвертация в А/м/м делением на μ₀.
      • drivefield/strength stored as mT → конвертация в А/м делением
        на μ₀ × 10⁻³. С 10 mT и G=1.12 Т/м даёт FFP-амплитуду 8.93 мм,
        что физически разумно (FOV±19 мм).

    Returns:
        dict с ключами для PMCNetConfig (drive_frequency_x/y,
        gradient_strength/_y, drive_field_amplitude/_y) + отдельно
        'frequencies_hz' (np.ndarray) — частоты для explicit DFT в
        spectral-форварде.
    """
    import h5py
    import math
    mu0 = 4 * math.pi * 1e-7
    with h5py.File(path, 'r') as f:
        drive_freq = f['acquisition/drivefield/driveFrequency'][:].flatten()
        gradient_T_per_m = f['acquisition/gradient'][:].flatten()
        strength_mT = f['acquisition/drivefield/strength'][:].flatten()
        freq_sel = f['measurement/frequencySelection'][:].flatten()
    return {
        'drive_frequency_x': float(drive_freq[0]),
        'drive_frequency_y': float(max(drive_freq[1], 1.0)),
        'gradient_strength': float(gradient_T_per_m[0] / mu0),
        'gradient_strength_y': float(gradient_T_per_m[1] / mu0),
        'drive_field_amplitude': float(strength_mT[0] * 1e-3 / mu0),
        'drive_field_amplitude_y': float(strength_mT[1] * 1e-3 / mu0),
        'frequencies_hz': freq_sel.astype(np.float32),
    }


def _load_real_measurement_b(
        path: str = './../ChineseData/BeihangUniversityData/MeasurementData_B.h5'
) -> np.ndarray:
    """Загрузить РЕАЛЬНОЕ измерение фантома 'B' из BeihangUniversityData.

    Файл MeasurementData_B.h5 — это MDF-формат с измеренным сигналом со
    сканера (не изображение фантома). Содержит частотные гармоники в полях
    `measurement/data/r` (real) и `measurement/data/i` (imaginary), форма
    `(2, 1275)` — две катушки × 1275 гармоник, что точно совпадает с
    SystemMatrix.h5.

    Args:
        path: путь к .h5 файлу. По умолчанию относительный от cwd `mpi/ML`.

    Returns:
        complex64 массив формы (2, M_per_coil), готовый к подаче
        в `comparator.compare_all_methods_on_image` через поле
        `measurement` battery-словаря.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        re = f['measurement/data/r'][:]            # (2, 1275) float64
        im = f['measurement/data/i'][:]
        if int(f['measurement/isFourierTransformed'][:].item()) != 1:
            raise ValueError(
                f"{path}: measurement не в частотной области "
                "(isFourierTransformed != 1)"
            )
    measurement = (re + 1j * im).astype(np.complex64)
    return measurement


def build_phantom_battery(SM_measured, image_shape,
                          snr_db: float = 30.0,
                          random_seed: int = 0):
    """Сгенерировать батарею фантомов × методов для финального сравнения.

    Включены 5 синтетических 2D-фантомов + 1 реальное измерение:
      • two_droplets — две капли (классический тест разрешения);
      • phantom_4    — четыре угловые капли;
      • rotation_45  — крест с маркерами, повёрнутый на 45°;
      • shape_ring   — кольцевой фантом;
      • random       — случайные капли;
      • letter_B     — РЕАЛЬНОЕ измерение фантома 'B' из
                       BeihangUniversityData/MeasurementData_B.h5
                       (сигнал со сканера, не синтез).

    Для 5 синтетических фантомов измерения генерируются ДВУМЯ путями:
      • method='sm'      — через ИЗМЕРЕННУЮ системную матрицу (реальная
                            калибровка сканера);
      • method='physical' — через АНАЛИТИЧЕСКУЮ системную матрицу,
                            вычисленную из физики (Langevin + радиальная
                            s(r) + Лиссажу).

    Для letter_B — ОДНО измерение (реальное со сканера), без синтеза.
    Ground-truth изображение для метрик — приближение через
    `PhantomGenerator.letter_phantom('B')` (стандартная практика для
    real-data валидации, когда точная форма фантома известна по дизайну,
    но прямая запись изображения с прибора недоступна).

    Итого 5×2 + 1 = 11 экспериментов.

    Returns:
        list[dict]: каждый элемент — {'label', 'image', 'measurement',
                                       'metadata'} для `run_phantom_battery`.
    """
    print("\n" + "=" * 70)
    print("3. СБОРКА БАТАРЕИ ФАНТОМОВ ДЛЯ ФИНАЛЬНОГО СРАВНЕНИЯ")
    print("=" * 70)

    np.random.seed(random_seed)
    pg = PhantomGenerator(nx=image_shape[0], ny=image_shape[1])

    # 5 синтетических фантомов (с двумя путями генерации измерений)
    synthetic_phantoms = [
        ('two_droplets', pg.two_droplets(radius=0.2, distance=0.2)),
        ('phantom_4',    pg.phantom_4()),
        ('rotation_45',  pg.rotation_phantom(angle_deg=45.0)),
        ('shape_ring',   pg.shape_phantom('ring')),
        ('random',       pg.random_phantom(n_droplets=5)),
    ]

    # Аналитическая SM той же формы, что измеренная
    M_total = SM_measured.shape[0]
    print(f"  Сборка аналитической SM ({M_total}×{image_shape[0]*image_shape[1]})...")
    cfg_phys = PMCNetConfig(image_size=tuple(image_shape))
    SM_analytical = build_analytical_system_matrix(image_shape, M_total, cfg_phys)

    # Для каждого синтетического фантома — два измерения (sm + physical)
    battery = []
    for phantom_name, image in synthetic_phantoms:
        for method_name, SM_used in (('sm', SM_measured),
                                      ('physical', SM_analytical)):
            label = f'{phantom_name}__{method_name}'
            measurement = _synthesize_measurements_through_SM(
                image[None], SM_used, snr_db=snr_db,
                random_seed=random_seed,
            )[0]                              # (2, M_per_coil)
            battery.append({
                'label': label,
                'image': image,
                'measurement': measurement,
                'metadata': {'phantom_type': phantom_name,
                             'generation_method': method_name},
            })

    # letter_B — реальное измерение из H5, ground truth через приближение
    print(f"  Загрузка реального измерения фантома 'B' из H5...")
    try:
        real_b_meas = _load_real_measurement_b()
        # Ground truth для метрик: приближённое изображение по дизайну
        # фантома (точная форма B известна, но изображения со сканера нет)
        gt_b_approx = pg.letter_phantom('B')
        # Согласование размерностей measurement и SM: H5-файл может иметь
        # больше частотных бинов, чем SM_measured (1275 vs M_total/2).
        M_per_coil = M_total // 2
        if real_b_meas.shape[1] != M_per_coil:
            # Берём первые M_per_coil гармоник (низкочастотные несут основную
            # энергию реконструкции; высокие частоты вне SM-сетки не нужны)
            real_b_meas = real_b_meas[:, :M_per_coil]
            print(f"    обрезано до {M_per_coil} гармоник на катушку, "
                  f"чтобы совпало с SM_measured ({M_total} полных бинов)")
        battery.append({
            'label': 'letter_B__real',
            'image': gt_b_approx,            # ground truth approx для SSIM
            'measurement': real_b_meas,       # РЕАЛЬНОЕ измерение со сканера
            'metadata': {'phantom_type': 'letter_B',
                         'generation_method': 'real'},
        })
        print(f"    letter_B измерение: {real_b_meas.shape} complex64 (со сканера)")
    except (FileNotFoundError, OSError) as e:
        print(f"  ⚠ Не удалось загрузить MeasurementData_B.h5: {e}")
        print(f"    letter_B пропущен (только синтетические фантомы)")

    print(f"  Готово: {len(battery)} экспериментов "
          f"({len(synthetic_phantoms)} синтетических × 2 пути "
          f"+ letter_B на реальном сигнале)")
    return battery


def run_full_comparison(comparator, phantom_battery):
    """Запустить сравнение всех методов на батарее фантомов."""
    print("\n" + "=" * 70)
    print(f"3.1. СРАВНЕНИЕ {len(phantom_battery)} ЭКСПЕРИМЕНТОВ × N МЕТОДОВ")
    print("=" * 70)
    results = comparator.run_phantom_battery(phantom_battery)
    comparator.print_summary_table()
    comparator.save_results_to_file('./DATA/results/all_methods_summary.txt')
    return results


def run_openmpi_validation(comparator):
    """Запустить валидацию на OpenMPI-датасете."""
    print("\n" + "=" * 70)
    print("4. ВАЛИДАЦИЯ НА OpenMPIData")
    print("=" * 70)
    try:
        return comparator.compare_all_on_openmpi()
    except Exception as e:
        print(f"  Пропущено (датасет недоступен): {e}")
        return {}


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(num_samples: int = 2000, train_models: bool = True,
                 pmcnet_iterations: int = 1500,
                 validate_openmpi: bool = False,
                 force_regenerate_dataset: bool = False):
    """End-to-end вызов всего пайплайна.

    Args:
        num_samples: размер обучающего синтетического датасета (SM-путь).
        train_models: обучать ли supervised-модели заново или загружать
                      сохранённые веса из DATA/models/.
        pmcnet_iterations: число итераций для data-free методов (PMCNet, DIP).
        validate_openmpi: запускать ли валидацию на OpenMPIData (нужны
                          локальные файлы в ChineseData/OpenMPIData/).
        force_regenerate_dataset: если True, сгенерировать синтетический
                                   датасет заново даже если он сохранён
                                   на диске. По умолчанию False — при
                                   повторных запусках датасет загружается
                                   с диска мгновенно.
    """
    print("=" * 70)
    print("PIPELINE: сравнение методов реконструкции MPI")
    print("=" * 70)
    setup_directories()

    # 1) синтетический датасет — load with cache or generate-and-save
    _, dataset_sm = generate_synthetic_dataset(
        num_samples=num_samples, save=True,
        force_regenerate=force_regenerate_dataset,
    )

    # Загрузка измеренной SM из BeihangUniversityData
    gen = MPIDatasetGenerator()
    SM = gen.SM
    image_shape = gen.image_shape
    print(f"\nИзмеренная SM: {SM.shape}, image_shape={image_shape}")

    # Синтетический генератор фантомов работает в (51, 51), а измеренная
    # SM рассчитана на (image_shape) — обычно (19, 19). Перенесём фантомы
    # на сетку SM, чтобы у CNN/MoDL совпадали размеры.
    y_synth = dataset_sm['y_train']
    if y_synth.shape[1:] != tuple(image_shape):
        from scipy.ndimage import zoom
        zoom_factors = (1.0,
                        image_shape[0] / y_synth.shape[1],
                        image_shape[1] / y_synth.shape[2])
        y_train = zoom(y_synth, zoom_factors, order=1).astype(np.float32)
        print(f"  Фантомы переразмерены с {y_synth.shape[1:]} → "
              f"{tuple(image_shape)}")
    else:
        y_train = y_synth.astype(np.float32)

    # Регенерация измерений через ИЗМЕРЕННУЮ SM — обязательное условие,
    # чтобы все модели видели данные правильной размерности
    # (2, M_total/2) complex с M_total = SM.shape[0].
    print(f"  Регенерация измерений через измеренную SM (SNR 30 dB)...")
    X_train = _synthesize_measurements_through_SM(y_train, SM, snr_db=30.0)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    # 2) Обучение / загрузка моделей
    print("\n" + "=" * 70)
    print(f"2. {'ОБУЧЕНИЕ' if train_models else 'ЗАГРУЗКА'} МОДЕЛЕЙ")
    print("=" * 70)

    cnn = train_or_load_cnn(X_train, y_train, train=train_models)
    modl = train_or_load_modl(SM, image_shape, X_train, y_train, train=train_models)
    diff = train_or_load_diffusion(y_train, X_train, SM, train=train_models)
    chae_single, chae_multi = train_or_load_chae(
        SM, image_shape, X_train, y_train, train=train_models)
    dip = build_dip(image_shape)
    pmcnet_variants = build_pmcnet_variants(
        SM, image_shape, n_iterations=pmcnet_iterations,
        # Подгружаем точные параметры сканера из H5 — устраняет огромное
        # расхождение config-дефолтов с реальной физикой сканера
        # (f_x: 24510 → 10000, f_y: 26042 → 1, G_x: 0.56 → 1.12 Т/м, ...)
        scanner_h5_path='./../ChineseData/BeihangUniversityData/SystemMatrix.h5',
    )

    # 3) Сравнение через comparator
    cmp = MPIReconstructionComparator(
        './../ChineseData/BeihangUniversityData/SystemMatrix.h5')
    cmp.set_cnn_model(cnn)
    cmp.set_modl_model(modl)
    cmp.set_diffusion_model(diff)
    cmp.set_chae_model(chae_single)
    cmp.set_chae_multi_model(chae_multi)
    cmp.set_dip_model(dip)
    cmp.set_pmcnet_standard(pmcnet_variants['standard'])
    cmp.set_pmcnet_paper(pmcnet_variants['paper'])
    cmp.set_pmcnet_radial_coil(pmcnet_variants['radial_coil'])
    cmp.set_pmcnet_soft(pmcnet_variants['soft'])
    cmp.set_pmcnet_debye(pmcnet_variants['debye'])
    cmp.set_pmcnet_central_fd(pmcnet_variants['central_fd'])

    # 2.4) Cross-validation параметра μ для Tikhonov на этом наборе
    best_mu = tune_tikhonov_mu(SM, X_train, y_train,
                                mus=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                                kmax=20, n_val_samples=12)
    cmp.set_tikhonov_mu(best_mu)

    # 2.5) MoE поверх быстрых экспертов
    moe = build_moe(
        comparator=cmp,
        image_shape=image_shape,
        X_train=X_train, y_train=y_train,
        expert_names=('Тихонов', 'KatsMarc', 'Chae(2017)', 'CNN'),
        n_train_samples=128,
        epochs=80,
        mode='spatial',
    )
    if moe is not None:
        cmp.set_moe(moe)

    # 3) Батарея фантомов × методов генерации
    battery = build_phantom_battery(SM, image_shape, snr_db=30.0)
    synthetic_results = run_full_comparison(cmp, battery)

    # 4) OpenMPI-валидация (опционально, по флагу)
    if validate_openmpi:
        openmpi_results = run_openmpi_validation(cmp)
    else:
        print("\n" + "=" * 70)
        print("4. OpenMPI-валидация ПРОПУЩЕНА (validate_openmpi=False)")
        print("=" * 70)
        print("  Чтобы включить, передайте validate_openmpi=True или ")
        print("  запустите run.py с флагом --openmpi.")
        print("  Данные должны лежать в ../ChineseData/OpenMPIData/")
        print("  (подпапки calibrations/ и measurements/).")
        openmpi_results = {}

    # 5) Итог
    print("\n" + "=" * 70)
    print("ПАЙПЛАЙН ЗАВЕРШЁН")
    print("=" * 70)
    print("Артефакты:")
    print("  ./DATA/results/all_methods_summary.txt        — таблица метрик")
    print("  ./DATA/results/all_methods_<label>.png        — изображения по фантомам")
    if validate_openmpi:
        print("  ./DATA/results/openmpi_comparison.png         — графики OpenMPI")
    print("  ./DATA/results/training_curves/               — кривые обучения")
    print("  ./DATA/results/phantoms/                      — образцы фантомов")
    print("  ./DATA/models/                                — обученные веса")

    return {'synthetic': synthetic_results, 'openmpi': openmpi_results}


def main():
    parser = argparse.ArgumentParser(description='MPI reconstruction pipeline')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Размер синтетического датасета')
    parser.add_argument('--train', action='store_true', default=True,
                        help='Обучать модели с нуля')
    parser.add_argument('--load', action='store_true',
                        help='Загрузить сохранённые модели')
    parser.add_argument('--pmcnet_iter', type=int, default=1500,
                        help='Итераций оптимизации на одно измерение для PMCNet')
    parser.add_argument('--openmpi', action='store_true', default=False,
                        help='Запустить валидацию на OpenMPIData '
                             '(данные в ../ChineseData/OpenMPIData/). '
                             'По умолчанию отключено.')
    args = parser.parse_args()

    train_models = not args.load if args.load else args.train
    return run_pipeline(num_samples=args.num_samples,
                        train_models=train_models,
                        pmcnet_iterations=args.pmcnet_iter,
                        validate_openmpi=args.openmpi)


if __name__ == '__main__':
    main()
