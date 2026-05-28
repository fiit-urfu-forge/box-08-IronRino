"""Финальный pipeline сравнения методов реконструкции MPI.

Шаги (вызываются из `run_pipeline`):

  1. Генерация синтетического датасета — оба пути из Chae (2017):
       • системная матрица (Chebyshev SM);
       • прямые физические уравнения (функция Ланжевена).
  2. Обучение/загрузка всех моделей:
       классические    — Tikhonov, Kaczmarz;
       по статьям      — Chae (single & multi-layer), DIP, Shang/FDS-MPI,
                         DEQ-MPI, PMCNet (Standard / Physics-Enhanced / Final);
       baseline'ы      — CNN, MoDL, DiffusionModel.
  3. Сравнение методов на двух-капельных фантомах (разные радиусы и
     расстояния) на основе ИЗМЕРЕННОЙ системной матрицы из
     BeihangUniversityData.
  4. Валидация на OpenMPIData-датасете.
  5. Сводные метрики (SSIM/PSNR/FWHM/время), визуализация, текстовый отчёт.

Все «по-настоящему обучаемые» модели поддерживают режимы train/load.
PMCNet и DIP — data-free, обучения не требуют. Tikhonov/Kaczmarz —
аналитические, без параметров.
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
    DeepImagePrior, ShangCNN, DEQMPI,
    PMCNetConfig, PMCNetStandard, PMCNetPaper,
    PMCNetPhysicsEnhanced, PMCNetFinal,
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


def generate_synthetic_dataset(num_samples: int = 2000, save: bool = True):
    """Сгенерировать датасет двумя путями + сохранить визуализацию фантомов.

    Возвращает: (generator, dataset_sm) — генератор для повторных вызовов
    и основной датасет (через системную матрицу).
    """
    print("\n" + "=" * 70)
    print("1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ (Chae 2017)")
    print("=" * 70)

    generator = SyntheticDatasetGenerator(nx=51, ny=51, n_harmonics=200)

    # Визуализация всех типов фантомов — для отчёта
    generator.phantom_gen.visualize_phantoms(
        save_path='./DATA/results/phantoms/all_phantoms.png',
    )

    # Путь 1: через системную матрицу (быстрее, более линейно)
    print(f"\n  Путь 1: SM-генерация, {num_samples} образцов...")
    dataset_sm = generator.create_training_pipeline_dataset(
        n_samples=num_samples, include_all_phantoms=True, save=save,
    )

    # Путь 2: через физические уравнения (для валидации/сравнения)
    print(f"\n  Путь 2: физические уравнения, {min(500, num_samples // 3)} образцов...")
    dataset_phys = generator.generate_dataset(
        n_samples=min(500, num_samples // 3),
        method='physical',
        phantom_types=[PhantomType.TWO_DROPLETS, PhantomType.PHANTOM_4,
                       PhantomType.CONCENTRATION, PhantomType.RESOLUTION,
                       PhantomType.SHAPE, PhantomType.PATTERN],
        particle_sizes_nm=[30, 40, 50],
        add_noise=True, snr_db=35.0, test_split=0.2,
    )
    if save:
        generator.save_dataset(dataset_phys, './DATA/dataset/synthetic_physical')

    print(f"\n  Готово: SM = {len(dataset_sm['X_train'])} train, "
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
    """Размер развёрнутого «комплекс → 4·M» вектора (Re/Im на 2 катушки)."""
    return X_train.shape[2] * 4


def _flatten_measurement_batch(X):
    """(N, 2, M) комплекс → (N, 4·M) вещественный."""
    real = X.real; imag = X.imag
    out = np.concatenate([real[:, 0], imag[:, 0],
                          real[:, 1], imag[:, 1]], axis=-1)
    return out.astype(np.float32)


def _synthesize_measurements_through_SM(y_images, SM,
                                        snr_db: float = 30.0,
                                        random_seed: int = 0):
    """Регенерация измерений через ИЗМЕРЕННУЮ системную матрицу.

    Синтетический генератор по умолчанию использует Chebyshev-SM с
    `n_harmonics=200`, что даёт X-шейпы, несовместимые с измеренной SM
    (которая обычно 2·n_freq_per_coil × N). Чтобы все обучаемые модели
    (CNN/MoDL/Chae/Shang/DEQ) видели данные правильной размерности,
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


def _format_for_modl_deq(X_complex):
    """(N, 2, M_per_coil) complex → (N, 4·M_per_coil) real.

    Для MoDL/DEQ — вход — расширенное (Re, Im) представление измерения
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
                      epochs: int = 40, output_size=(51, 51)):
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
                       train: bool = True, epochs: int = 30):
    print("\n  MoDL baseline...")
    path = './DATA/models/modl_best.pth'
    trainer = ModelTrainerFactory.create_modl_trainer(
        system_matrix=SM, image_shape=image_shape,
        n_iterations=3, lambda_param=0.01, base_filters=32,
    )
    if not train and os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("    загружена")
        return trainer

    X_real = _format_for_modl_deq(X_train)            # (N, 2·M_total)
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
                            train: bool = True, epochs: int = 25):
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
                       train: bool = True, epochs: int = 60):
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


# --- Shang 2022 FDS-MPI -----------------------------------------------------

def train_or_load_shang(X_train, y_train, SM, train: bool = True,
                        epochs: int = 10, tik_kmax: int = 5,
                        max_train_samples: int = 300):
    """[ВРЕМЕННО ОТКЛЮЧЁН в `run_pipeline`] FDS-MPI Shang 2022.

    Узкое место — Tikhonov-препроцессинг на каждый X_train (на 3000
    семплов с kmax=15 это ~3 с/семпл = 2.5 часа _до_ старта тренировки
    самой сети). Дефолты уменьшены, чтобы при включении функция
    отработала за разумное время:
      • epochs: 40 → 10
      • Tikhonov kmax: 15 → 5
      • Ограничиваем сабсет до 300 семплов через `max_train_samples`
        (Shang всё равно учит постпроцессинг, для него полный датасет
        избыточен).

    Чтобы включить обратно — раскомментируйте вызов в `run_pipeline`.
    """
    print("\n  Shang (2022) FDS-MPI dual-branch...")
    path = './DATA/models/shang_best.pth'
    dev = device()
    # base_filters=64 как в статье Shang 2022 (см. Sec. 3.2: "the number
    # of filters of the other layers was set to 64")
    model = ShangCNN(input_channels=1, output_channels=1, base_filters=64).to(dev)
    if not train and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=dev),
                              strict=False)
        print("    загружена")
        return model

    # Ограничиваем сабсет — Tikhonov-препроцессинг очень медленный
    n = min(max_train_samples, len(X_train))
    if n < len(X_train):
        idx = np.random.choice(len(X_train), n, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"    Сабсет {n}/{len(idx)} семплов для ускорения "
              f"Tikhonov-prep")

    # FDS-MPI — постпроцессинг: на вход даём грубую LOW-RES реконструкцию,
    # цель — точное y. В статье вход — это X-space-реконструкция при
    # слабом градиенте. Здесь используем Tikhonov как ближайший аналог
    # (гладкое регуляризованное решение).
    from .models.classical import TikhonovReconstructor
    tik = TikhonovReconstructor(SM)
    inputs = []
    for m in tqdm(X_train, desc='    Shang: prep Tikhonov inputs',
                  leave=False):
        v = np.concatenate([m[0], m[1]])
        recon = tik.reconstruct(v, mu=1e-2,
                                 kmax=tik_kmax).reshape(*y_train.shape[1:])
        inputs.append(recon)
    X_lr = np.array(inputs, dtype=np.float32)[:, None]
    y_hr = y_train.astype(np.float32)[:, None]
    ds = TensorDataset(torch.tensor(X_lr), torch.tensor(y_hr))
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for ep in tqdm(range(epochs), desc='    Shang'):
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
    _save_curve(losses, 'Shang FDS-MPI',
                './DATA/results/training_curves/shang_training.png')
    return model


# --- DEQ-MPI ----------------------------------------------------------------

def train_or_load_deq(SM, image_shape, X_train, y_train,
                      train: bool = True, epochs: int = 40):
    print("\n  DEQ-MPI (Güngör 2024) — RDN + LC...")
    path = './DATA/models/deq_best.pth'
    dev = device()
    # Архитектура строго из статьи Güngör 2024 Sec IV.A:
    # n_res=4, F_R=12, n_conv=12, n_LC=1, F_LC=8, ε=√M, 25 итер с Anderson.
    model = DEQMPI(system_matrix=SM, image_shape=image_shape,
                   n_iterations=25, lambda_param=1.0,
                   rdn_channels=12, n_rdn_modules=4,
                   rdn_convs_per_module=12,
                   lc_hidden=8, use_anderson=True).to(dev)
    if not train and os.path.exists(path):
        ckpt = torch.load(path, map_location=dev)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("    загружена")
        return model

    X_real = _format_for_modl_deq(X_train)
    y_img = y_train[:, None].astype(np.float32)

    # КРИТИЧНО (статья Sec V.A): pre-train RDN и LC отдельно перед
    # основным обучением. Без этого PSNR падает на 7–17 dB.
    # batch_size=4 — компромисс для GPU с 4 GB VRAM. Основной OOM-fix
    # реализован внутри DEQMPI.forward через implicit-DEQ gradient
    # (память O(1) по числу итераций); этот batch_size — дополнительный
    # запас для очень тесных GPU.
    deq_batch = 4
    pretrain_eps = max(5, epochs // 3)
    print(f"    [init 1/2] Pre-train RDN как denoiser ({pretrain_eps} эп., σ₁=0.1)...")
    y_img_t = torch.tensor(y_img)
    model.pretrain_rdn(y_img_t, sigma1=0.1, epochs=pretrain_eps, lr=1e-3,
                       batch_size=deq_batch)
    print(f"    [init 2/2] Pre-train LC как L2-проекция ({pretrain_eps} эп., σ₂=0.05, σ₃=0.02)...")
    # Чистые измерения y_clean = SM @ y_train (без шума)
    y_clean = np.zeros_like(X_real)
    A_np = model.A.cpu().numpy()
    for i in tqdm(range(len(y_img)), desc='    DEQ: prep y_clean = SM·c',
                  leave=False):
        c = y_img[i, 0].flatten()
        y_clean[i] = A_np @ c
    model.pretrain_lc(torch.tensor(y_clean), sigma2=0.05, sigma3=0.02,
                      epochs=pretrain_eps, lr=1e-3, batch_size=deq_batch)

    ds = TensorDataset(torch.tensor(X_real), torch.tensor(y_img))
    loader = DataLoader(ds, batch_size=deq_batch, shuffle=True)
    crit = torch.nn.L1Loss()  # статья: L1 на выходе
    opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                           betas=(0.9, 0.999))  # статья: Adam β=(0.9, 0.999)
    losses = []
    best = float('inf')
    for ep in tqdm(range(epochs), desc='    DEQ'):
        ep_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad()
            y_pred = model(xb)
            loss = crit(y_pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if avg < best:
            best = avg
            torch.save({'model_state_dict': model.state_dict()}, path)
    _save_curve(losses, 'DEQ-MPI',
                './DATA/results/training_curves/deq_training.png')
    return model


# --- PMCNet trio (data-free) -----------------------------------------------

def build_pmcnet_quartet(SM, image_shape,
                          n_iterations: int = 1500, n_colors: int = 2,
                          init_tau_seconds: float = 2.0e-6):
    """Собрать четыре варианта PMCNet (data-free, paper Sec. III.B).

    Returns:
        (standard, paper, physics_enhanced, final)
          standard: SM-baseline (не из paper), форвард = S_measured · c
          paper:    paper-faithful Huang 2026 (Eq. 1-3), базовая физика
          physics_enhanced: paper + радиальная p, центр. FD, langevin_safe
          final:    PhysicsEnhanced + Debye + multi-color + TV
    """
    print("\n  PMCNet (Huang 2026) — четыре варианта (data-free)...")
    base = PMCNetConfig(
        image_size=tuple(image_shape),
        n_iterations=n_iterations,
        learning_rate=1e-3,
    )
    n_meas = int(SM.shape[0])
    standard = PMCNetStandard(SM, image_shape, config=base)
    paper = PMCNetPaper(image_shape, n_meas, config=base)
    phys = PMCNetPhysicsEnhanced(image_shape, n_meas, config=base)
    final_cfg = PMCNetConfig(**{
        **base.__dict__,
        'n_colors': n_colors, 'use_debye': True,
        'init_tau_seconds': init_tau_seconds, 'lambda_tv': 1e-3,
    })
    final = PMCNetFinal(image_shape, n_meas,
                        config=final_cfg, n_colors=n_colors)
    print(f"    1) Standard          (SM-baseline — НЕ из paper)")
    print(f"    2) Paper             (paper-faithful, Huang 2026 Eq. 1-3)")
    print(f"    3) Physics-Enhanced  (paper + радиальная p, центр. FD, ...)")
    print(f"    4) Final             (PhysicsEnhanced + Debye + multi-color + TV)")
    return standard, paper, phys, final




# ---------------------------------------------------------------------------
# 2.5. Mixture of Experts — комбинирование моделей
# ---------------------------------------------------------------------------


def build_moe(comparator, image_shape,
              X_train, y_train,
              expert_names=('Тихонов', 'KatsMarc', 'Chae(2017)',
                            'Shang(2022)', 'CNN'),
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

    PMCNet намеренно НЕ включён в дефолтный список экспертов: его
    реконструкция требует ~1500 итераций оптимизации на одно
    измерение, и прекомпьют по нескольким десяткам образцов получится
    слишком долгим. Добавляйте `'PMCNet-Std(2026)'` и др. вручную,
    если есть бюджет времени.
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
        'DIP(2020)': comparator.dip_reconstruction,
        'Shang(2022)': comparator.shang_reconstruction,
        'DEQ-MPI(2024)': comparator.deq_reconstruction,
        'CNN': comparator.cnn_reconstruction,
        'MoDL': comparator.modl_reconstruction,
        'Diffusion': comparator.diffusion_reconstruction,
        'PMCNet-Std(2026)': comparator.pmcnet_standard_reconstruction,
        'PMCNet-Phys(2026)': comparator.pmcnet_physics_enhanced_reconstruction,
        'PMCNet-Final(2026)': comparator.pmcnet_final_reconstruction,
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


def build_phantom_battery(SM_measured, image_shape,
                          snr_db: float = 30.0,
                          random_seed: int = 0):
    """Сгенерировать батарею фантомов × методов для финального сравнения.

    Включены 6 типов 2D-фантомов:
      • two_droplets — две капли (классический тест разрешения);
      • phantom_4    — четыре угловые капли;
      • rotation_45  — крест с маркерами, повёрнутый на 45°;
      • shape_ring   — кольцевой фантом;
      • random       — случайные капли;
      • letter_B     — стилизованная буква B (приближение к
                       MeasurementData_B.h5 из BeihangUniversityData).

    Для каждого фантома измерения генерируются ДВУМЯ путями (как в
    статье Chae 2017):
      • method='sm'      — через ИЗМЕРЕННУЮ системную матрицу (реальная
                            калибровка сканера);
      • method='physical' — через АНАЛИТИЧЕСКУЮ системную матрицу,
                            вычисленную из физики (Langevin + радиальная
                            s(r) + Лиссажу). Имеет ту же форму, что и
                            измеренная — все модели работают одинаково.

    Итого 6 × 2 = 12 экспериментов.

    Returns:
        list[dict]: каждый элемент — {'label', 'image', 'measurement',
                                       'metadata'} для `run_phantom_battery`.
    """
    print("\n" + "=" * 70)
    print("3. СБОРКА БАТАРЕИ ФАНТОМОВ ДЛЯ ФИНАЛЬНОГО СРАВНЕНИЯ")
    print("=" * 70)

    np.random.seed(random_seed)
    pg = PhantomGenerator(nx=image_shape[0], ny=image_shape[1])

    # 6 типов фантомов (ground truth)
    phantoms = [
        ('two_droplets', pg.two_droplets(radius=0.2, distance=0.2)),
        ('phantom_4',    pg.phantom_4()),
        ('rotation_45',  pg.rotation_phantom(angle_deg=45.0)),
        ('shape_ring',   pg.shape_phantom('ring')),
        ('random',       pg.random_phantom(n_droplets=5)),
        ('letter_B',     pg.letter_phantom('B')),
    ]

    # Аналитическая SM той же формы, что измеренная
    M_total = SM_measured.shape[0]
    print(f"  Сборка аналитической SM ({M_total}×{image_shape[0]*image_shape[1]})...")
    cfg_phys = PMCNetConfig(image_size=tuple(image_shape))
    SM_analytical = build_analytical_system_matrix(image_shape, M_total, cfg_phys)

    # Для каждого фантома — два измерения (sm + physical)
    battery = []
    for phantom_name, image in phantoms:
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

    print(f"  Готово: {len(battery)} экспериментов "
          f"({len(phantoms)} фантомов × 2 метода)")
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
                 validate_openmpi: bool = False):
    """End-to-end вызов всего пайплайна."""
    print("=" * 70)
    print("PIPELINE: сравнение методов реконструкции MPI")
    print("=" * 70)
    setup_directories()

    # 1) синтетический датасет (оба пути из Chae 2017)
    _, dataset_sm = generate_synthetic_dataset(num_samples=num_samples, save=True)

    # Загрузка измеренной SM из BeihangUniversityData
    gen = MPIDatasetGenerator()
    SM = gen.SM
    image_shape = gen.image_shape
    print(f"\nИзмеренная SM: {SM.shape}, image_shape={image_shape}")

    # Синтетический генератор фантомов работает в (51, 51), а измеренная
    # SM рассчитана на (image_shape) — обычно (19, 19). Перенесём фантомы
    # на сетку SM, чтобы у CNN/MoDL/DEQ совпадали размеры.
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
    # Shang FDS-MPI временно отключён — Tikhonov-препроцессинг на
    # каждый X_train занимает ~3 с/семпл (на 3000 семплов = 2.5 часа
    # ДО старта тренировки сети). См. `train_or_load_shang` — там
    # дефолты уже уменьшены (epochs=10, kmax=5, max_train_samples=300).
    # Раскомментируйте обе строки ниже, чтобы включить обратно.
    # shang = train_or_load_shang(X_train, y_train, SM, train=train_models)
    shang = None
    deq = train_or_load_deq(SM, image_shape, X_train, y_train, train=train_models)
    pmcnet_std, pmcnet_paper, pmcnet_phys, pmcnet_final = build_pmcnet_quartet(
        SM, image_shape, n_iterations=pmcnet_iterations, n_colors=2,
    )

    # 3) Сравнение через comparator
    cmp = MPIReconstructionComparator(
        './../ChineseData/BeihangUniversityData/SystemMatrix.h5')
    cmp.set_cnn_model(cnn)
    cmp.set_modl_model(modl)
    cmp.set_diffusion_model(diff)
    cmp.set_chae_model(chae_single)
    cmp.set_dip_model(dip)
    if shang is not None:
        cmp.set_shang_model(shang)
    cmp.set_deq_model(deq)
    cmp.set_pmcnet_standard(pmcnet_std)
    cmp.set_pmcnet_paper(pmcnet_paper)
    cmp.set_pmcnet_physics_enhanced(pmcnet_phys)
    cmp.set_pmcnet_final(pmcnet_final)

    # 2.4) Cross-validation параметра μ для Tikhonov на этом наборе
    best_mu = tune_tikhonov_mu(SM, X_train, y_train,
                                mus=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                                kmax=20, n_val_samples=12)
    cmp.set_tikhonov_mu(best_mu)

    # 2.5) MoE поверх быстрых экспертов (Shang исключён — отключён выше)
    moe_experts = ['Тихонов', 'KatsMarc', 'Chae(2017)', 'CNN']
    if shang is not None:
        moe_experts.insert(3, 'Shang(2022)')
    moe = build_moe(
        comparator=cmp,
        image_shape=image_shape,
        X_train=X_train, y_train=y_train,
        expert_names=tuple(moe_experts),
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
