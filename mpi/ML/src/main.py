# main.py (полная версия с интегрированной генерацией синтетических данных)

"""Главный модуль для запуска пайплайна со всеми моделями"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .trainer import ModelTrainerFactory, MPITrainer
from .comparator import MPIReconstructionComparator
from .models import (
    ChaeSingleLayerNN, DeepImagePrior, ShangCNN, PGNet, DEQMPI,
    KatsMarcAlgorithm, TikhonovReconstructor
)
from .pmcnet import (
    PMCNetConfig,
    PMCNetReconstructor, PMCNetRefinedReconstructor,
    PMCNetStandard, PMCNetPhysicsEnhanced, PMCNetFinal,
)
from .visualization import Visualization
from .openmpi_loader import OpenMPIDataManager, OpenMPIDataset
from .synthetic_data_generator import (
    SyntheticDatasetGenerator, PhantomType, NanoparticleProperties,
    ChebyshevSystemFunction, PhysicalMPISimulator, PhantomGenerator
)


class MoDLDataset:
    """Dataset для MoDL модели"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        meas = self.X[idx]
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        target = self.y[idx]
        return (torch.tensor(meas_vector, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


class DiffusionDataset:
    """Dataset для диффузионной модели"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        meas = self.X[idx]
        target = self.y[idx]

        # Преобразование измерений в 4-канальное изображение
        real_part = meas.real
        imag_part = meas.imag
        combined = np.concatenate([real_part, imag_part], axis=0)

        # Изменение размера до квадратной формы
        n_measurements = combined.shape[1]
        size = int(np.sqrt(n_measurements))
        if size * size != n_measurements:
            size += 1
            target_size = size * size
            padded = np.zeros((4, target_size), dtype=np.float32)
            padded[:, :n_measurements] = combined
            combined = padded

        combined = combined.reshape(4, size, size)

        # Интерполяция до нужного размера
        from scipy import ndimage
        if combined.shape[1] != 51:
            combined = ndimage.zoom(combined, (1, 51 / combined.shape[1], 51 / combined.shape[2]), order=1)

        return (torch.tensor(combined, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


def create_directories():
    """Создание необходимых директорий"""
    os.makedirs('./DATA/dataset', exist_ok=True)
    os.makedirs('./DATA/models', exist_ok=True)
    os.makedirs('./DATA/results', exist_ok=True)
    os.makedirs('./DATA/results/training_curves', exist_ok=True)
    os.makedirs('./DATA/results/phantoms', exist_ok=True)


def generate_synthetic_dataset_advanced(num_samples=5000):
    """
    Расширенная генерация синтетических данных
    с использованием физических уравнений Chae (2017)

    Включает:
    - Все типы фантомов (concentration, resolution, rotation, shape и др.)
    - Физические уравнения из работы Chae 2017
    - Разные размеры наночастиц (20-60 nm)
    - Два метода генерации (системная матрица и физические уравнения)
    """
    print("\n" + "="*70)
    print("1a. РАСШИРЕННАЯ ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("    (Chae 2017 физические уравнения + все типы фантомов)")
    print("="*70)

    # Создание генератора с параметрами из Chae 2017
    generator = SyntheticDatasetGenerator(nx=51, ny=51, n_harmonics=200)

    # 1. Визуализация всех типов фантомов для обучения
    print("\n  Генерация визуализации фантомов...")
    generator.phantom_gen.visualize_phantoms(
        save_path='./DATA/results/phantoms/all_phantoms.png'
    )

    # 2. Анализ влияния размера частиц (Chae 2017, Fig. 5, Fig. 9)
    print("\n  Анализ влияния размера наночастиц...")
    generator.analyze_particle_size_effect(
        particle_sizes=[20, 30, 35, 40, 50, 60],
        save_path='./DATA/results/phantoms/particle_size_analysis.png'
    )

    # 3. Сравнение методов генерации
    print("\n  Сравнение методов генерации (системная матрица vs физические уравнения)...")
    dataset_sm, dataset_phys = generator.compare_generation_methods(
        n_samples=100,
        save_path='./DATA/results/phantoms/methods_comparison.png'
    )

    # 4. Создание основного датасета для обучения (системная матрица - быстрее)
    print(f"\n  Создание основного датасета из {num_samples} образцов...")
    dataset_main = generator.create_training_pipeline_dataset(
        n_samples=num_samples,
        include_all_phantoms=True,
        save=True
    )

    # 5. Создание дополнительного датасета с физическими уравнениями
    print(f"\n  Создание физического датасета (для валидации)...")
    dataset_physical_full = generator.generate_dataset(
        n_samples=min(2000, num_samples // 3),
        method='physical',
        phantom_types=[PhantomType.TWO_DROPLETS, PhantomType.PHANTOM_4,
                      PhantomType.CONCENTRATION, PhantomType.RESOLUTION,
                      PhantomType.SHAPE, PhantomType.PATTERN],
        particle_sizes_nm=[30, 40, 50],
        add_noise=True,
        snr_db=35.0,
        test_split=0.2
    )
    generator.save_dataset(dataset_physical_full, './DATA/dataset/synthetic_physical')

    # 6. Создание тестового датасета для валидации разрешения
    print("\n  Создание тестового датасета для валидации разрешения...")
    resolution_test = create_resolution_test_dataset(generator)

    # 7. Генерация статистики по датасету
    print("\n  Статистика созданных датасетов:")
    print(f"    Основной датасет (системная матрица):")
    print(f"      - Обучающая выборка: {len(dataset_main['X_train'])}")
    print(f"      - Тестовая выборка: {len(dataset_main['X_test'])}")
    print(f"      - Типы фантомов: {len(set(m['phantom_type'] for m in dataset_main['metadata_train']))}")

    print(f"\n    Физический датасет:")
    print(f"      - Обучающая выборка: {len(dataset_physical_full['X_train'])}")
    print(f"      - Тестовая выборка: {len(dataset_physical_full['X_test'])}")
    print(f"      - Размеры частиц: {set(m['particle_size_nm'] for m in dataset_physical_full['metadata_train'])}")

    return generator, {
        'main': dataset_main,
        'physical': dataset_physical_full,
        'resolution_test': resolution_test
    }


def create_resolution_test_dataset(generator, n_configs=10):
    """
    Создание тестового датасета для оценки разрешения
    Разные расстояния между каплями для измерения FWHM
    """
    distances = np.linspace(0.05, 0.4, n_configs)
    test_images = []
    test_measurements = []
    test_metadata = []

    for dist in distances:
        # Генерация фантома с двумя каплями
        image = generator.phantom_gen.two_droplets(radius=0.1, distance=dist)

        # Генерация измерений для разных размеров частиц
        for size in [30, 40, 50]:
            props = NanoparticleProperties(diameter_nm=size)
            measurement = generator.phys_sim.generate_measurement(image, props, add_noise=True, snr_db=40.0)

            test_images.append(image)
            test_measurements.append(measurement)
            test_metadata.append({
                'distance': dist,
                'particle_size_nm': size,
                'theoretical_resolution_mm': props.langevin_fwhm_mm
            })

    # Сохранение
    np.save('./DATA/dataset/resolution_test_X.npy', np.array(test_measurements))
    np.save('./DATA/dataset/resolution_test_y.npy', np.array(test_images))

    import json
    with open('./DATA/dataset/resolution_test_metadata.json', 'w') as f:
        json.dump(test_metadata, f, indent=2)

    print(f"    Разрешающий тест: {len(test_images)} образцов")
    print(f"    Расстояния: {distances}")

    return {
        'X': np.array(test_measurements),
        'y': np.array(test_images),
        'metadata': test_metadata
    }


def generate_dataset(num_samples=5000):
    """Генерация базового датасета (оригинальный метод)"""
    print("\n" + "="*70)
    print("1. БАЗОВАЯ ГЕНЕРАЦИЯ ДАТАСЕТА")
    print("="*70)
    generator = MPIDatasetGenerator()
    generator.create_dataset(num_samples=num_samples)
    return generator


def train_cnn_model():
    """Обучение улучшенной CNN модели"""
    print("\n" + "=" * 70)
    print("2. ОБУЧЕНИЕ CNN МОДЕЛИ")
    print("=" * 70)

    # Проверяем наличие расширенного датасета
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
        X_test = np.load('./DATA/dataset/synthetic_complete_X_test.npy')
        y_test = np.load('./DATA/dataset/synthetic_complete_y_test.npy')

        # Создание dataset
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train[:, np.newaxis, :, :], dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test[:, np.newaxis, :, :], dtype=torch.float32)
        )
    else:
        # Использование базового датасета
        train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
        test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    sample_measurement, sample_target = train_dataset[0]
    print(f"  Размер входных данных: {sample_measurement.shape}")
    print(f"  Размер выходных данных: {sample_target.shape}")

    # Проверка согласованности размеров
    if isinstance(sample_measurement, torch.Tensor):
        n_channels = sample_measurement.shape[0]
    else:
        n_channels = sample_measurement.shape[0]

    assert n_channels == 4, f"Expected 4 input channels, got {n_channels}"
    print(f"  ✓ Входные каналы: {n_channels}")

    # Используем меньший batch size для стабильности
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = ModelTrainerFactory.create_cnn_trainer(
        input_channels=4,
        output_channels=1,
        learning_rate=1e-3,
        base_filters=32
    )

    # Обучаем
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_path='./DATA/models/cnn_best.pth'
    )

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('CNN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/cnn_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ CNN модель обучена")
    return trainer


def train_modl_model(system_matrix, image_shape):
    """Обучение улучшенной MoDL модели"""
    print("\n" + "=" * 70)
    print("3. ОБУЧЕНИЕ MoDL МОДЕЛИ")
    print("=" * 70)

    # Проверяем наличие расширенного датасета
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем только часть данных для ускорения
    n_samples = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    split_idx = int(0.8 * len(X_train))
    train_dataset = MoDLDataset(X_train[:split_idx], y_train[:split_idx])
    val_dataset = MoDLDataset(X_train[split_idx:], y_train[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    trainer = ModelTrainerFactory.create_modl_trainer(
        system_matrix=system_matrix,
        image_shape=image_shape,
        n_iterations=3,
        lambda_param=0.01,
        learning_rate=1e-3,
        base_filters=32
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=20,
        save_path='./DATA/models/modl_best.pth'
    )

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('MoDL Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/modl_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ MoDL модель обучена")
    return trainer


def train_diffusion_model():
    """Обучение улучшенной диффузионной модели"""
    print("\n" + "=" * 70)
    print("4. ОБУЧЕНИЕ DIFFUSION МОДЕЛИ")
    print("=" * 70)

    # Проверяем наличие расширенного датасета
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')

    # Используем меньше данных для ускорения
    n_samples = min(500, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    X_train_prepared = []
    y_train_prepared = []

    print("  Подготовка данных...")
    for i in tqdm(range(len(X_train)), desc="  Подготовка"):
        meas = X_train[i]
        target = y_train[i]

        # Преобразование измерений в 4-канальное изображение
        real_part = meas.real
        imag_part = meas.imag
        combined = np.concatenate([real_part, imag_part], axis=0)

        # Изменение размера до квадратной формы
        n_measurements = combined.shape[1]
        size = int(np.ceil(np.sqrt(n_measurements)))
        target_size = size * size
        padded = np.zeros((4, target_size), dtype=np.float32)
        padded[:, :n_measurements] = combined
        combined = padded.reshape(4, size, size)

        # Изменение размера до 51x51
        from scipy import ndimage
        if combined.shape[1] != 51:
            combined = ndimage.zoom(combined, (1, 51 / combined.shape[1], 51 / combined.shape[2]), order=1)

        X_train_prepared.append(combined)
        y_train_prepared.append(target)

    X_train_prepared = np.array(X_train_prepared, dtype=np.float32)
    y_train_prepared = np.array(y_train_prepared, dtype=np.float32)

    # Добавляем канал для target
    y_train_prepared = y_train_prepared[:, np.newaxis, :, :]

    print(f"  X_train_prepared shape: {X_train_prepared.shape}")
    print(f"  y_train_prepared shape: {y_train_prepared.shape}")

    # Разделение на train/val
    split_idx = int(0.8 * len(X_train_prepared))

    train_dataset = TensorDataset(
        torch.tensor(X_train_prepared[:split_idx], dtype=torch.float32),
        torch.tensor(y_train_prepared[:split_idx], dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_train_prepared[split_idx:], dtype=torch.float32),
        torch.tensor(y_train_prepared[split_idx:], dtype=torch.float32)
    )

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Параметры модели
    n_steps = 100
    base_filters = 64
    beta_start = 1e-4
    beta_end = 0.02

    trainer = ModelTrainerFactory.create_diffusion_trainer(
        n_steps=n_steps,
        learning_rate=1e-4,
        image_size=51,
        base_filters=base_filters,
        beta_start=beta_start,
        beta_end=beta_end
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=10,
        save_path='./DATA/models/diffusion_best.pth'
    )

    # После обучения сохраняем параметры модели
    checkpoint = torch.load('./DATA/models/diffusion_best.pth')
    checkpoint['n_steps'] = n_steps
    checkpoint['base_filters'] = base_filters
    checkpoint['beta_start'] = beta_start
    checkpoint['beta_end'] = beta_end
    torch.save(checkpoint, './DATA/models/diffusion_best.pth')

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Diffusion Model Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/diffusion_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Diffusion модель обучена (n_steps={n_steps})")
    return trainer


def train_chae_model(system_matrix, image_shape):
    """Обучение модели Chae (2017) с батчевой обработкой"""
    print("\n" + "=" * 70)
    print("5. ОБУЧЕНИЕ CHAE МОДЕЛИ (2017) - БАТЧЕВАЯ ВЕРСИЯ")
    print("=" * 70)

    # Загрузка данных (приоритет расширенному датасету)
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
        X_test = np.load('./DATA/dataset/synthetic_complete_X_test.npy')
        y_test = np.load('./DATA/dataset/synthetic_complete_y_test.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')
        X_test = np.load('./DATA/dataset/X_test.npy')
        y_test = np.load('./DATA/dataset/y_test.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Параметры модели
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 5100
    output_dim = image_shape[0] * image_shape[1]  # 2601

    print(f"  input_dim = {input_dim}")
    print(f"  output_dim = {output_dim}")

    # Векторизованная подготовка данных
    print("  Подготовка данных (векторизовано)...")

    # Разделяем реальные и мнимые части
    real_parts = X_train.real
    imag_parts = X_train.imag

    # Объединяем в один вектор
    X_train_vec = np.concatenate([
        real_parts[:, 0, :],
        imag_parts[:, 0, :],
        real_parts[:, 1, :],
        imag_parts[:, 1, :]
    ], axis=1)

    # Нормализация входных данных
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_vec = scaler.fit_transform(X_train_vec)

    # Подготовка выходных данных
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    # Подготовка тестовых данных
    real_parts_test = X_test.real
    imag_parts_test = X_test.imag
    X_test_vec = np.concatenate([
        real_parts_test[:, 0, :],
        imag_parts_test[:, 0, :],
        real_parts_test[:, 1, :],
        imag_parts_test[:, 1, :]
    ], axis=1)
    X_test_vec = scaler.transform(X_test_vec)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_flat shape: {y_train_flat.shape}")

    # Создание DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_vec, dtype=torch.float32),
        torch.tensor(y_test_flat, dtype=torch.float32)
    )

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    model = ChaeSingleLayerNN(
        input_dim,
        output_dim,
        hidden_dim=1024,
        dropout_rate=0.2,
        use_batch_norm=True
    )
    model = model.to(device)

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    # Планировщик обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Функция потерь
    criterion = nn.MSELoss()

    # Ранняя остановка
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    print("  Начало батчевого обучения Chae модели...")
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(100), desc="Обучение Chae (батчевое)")
    for epoch in epoch_pbar:
        # Обучение
        model.train()
        total_train_loss = 0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Валидация
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 1024,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_
            }, './DATA/models/chae_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n  Ранняя остановка на эпохе {epoch}")
            break

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}',
            'best': f'{best_val_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    # Сохранение scaler
    np.save('./DATA/models/chae_scaler_mean.npy', scaler.mean_)
    np.save('./DATA/models/chae_scaler_scale.npy', scaler.scale_)

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Chae Model Training Progress (Batch Training)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('./DATA/results/training_curves/chae_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Chae модель обучена (батчевая версия)")
    print(f"  Лучшая loss: {best_val_loss:.6f}")

    # Загрузка лучшей модели
    checkpoint = torch.load('./DATA/models/chae_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def train_shang_model():
    """Обучение модели Shang et al. (2020)"""
    print("\n" + "=" * 70)
    print("6. ОБУЧЕНИЕ SHANG МОДЕЛИ (2020)")
    print("=" * 70)

    # Загрузка данных
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
        X_test = np.load('./DATA/dataset/synthetic_complete_X_test.npy')
        y_test = np.load('./DATA/dataset/synthetic_complete_y_test.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')
        X_test = np.load('./DATA/dataset/X_test.npy')
        y_test = np.load('./DATA/dataset/y_test.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    # Загружаем системную матрицу
    SM = np.load('./DATA/dataset/SM.npy')
    print(f"  SM shape: {SM.shape}")

    # Используем ограниченное количество образцов
    n_samples = min(500, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Создаем начальные реконструкции через псевдообращение
    print("  Вычисление начальных реконструкций...")

    from scipy.linalg import pinv
    A_pinv = pinv(SM)

    X_train_images = []
    for i in tqdm(range(n_samples), desc="  Подготовка изображений"):
        meas = X_train[i]
        real_part = meas.real
        meas_vector = real_part.flatten()
        recon = A_pinv @ meas_vector
        recon = recon.reshape(51, 51)

        if recon.max() > recon.min():
            recon = (recon - recon.min()) / (recon.max() - recon.min())
        else:
            recon = np.zeros_like(recon)

        X_train_images.append(recon)

    X_train_images = np.array(X_train_images)[:, np.newaxis, :, :].astype(np.float32)

    # Подготовка тестовых данных
    n_test_samples = min(100, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    X_test_images = []
    for i in tqdm(range(n_test_samples), desc="  Подготовка тестовых изображений"):
        meas = X_test[i]
        real_part = meas.real
        meas_vector = real_part.flatten()
        recon = A_pinv @ meas_vector
        recon = recon.reshape(51, 51)

        if recon.max() > recon.min():
            recon = (recon - recon.min()) / (recon.max() - recon.min())
        else:
            recon = np.zeros_like(recon)

        X_test_images.append(recon)

    X_test_images = np.array(X_test_images)[:, np.newaxis, :, :].astype(np.float32)

    # Целевые изображения
    y_train_images = y_train[:n_samples, np.newaxis, :, :].astype(np.float32)
    y_test_images = y_test[:n_test_samples, np.newaxis, :, :].astype(np.float32)

    # DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_images, dtype=torch.float32),
        torch.tensor(y_train_images, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_images, dtype=torch.float32),
        torch.tensor(y_test_images, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Создание модели
    model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)
    model = model.to(device)

    # Оптимизатор
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("  Начало обучения Shang модели...")
    model.train()
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(20), desc="Обучение Shang")
    for epoch in epoch_pbar:
        # Обучение
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}'
        })

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, './DATA/models/shang_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Shang Model Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('./DATA/results/training_curves/shang_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Shang модель обучена, лучшая loss: {best_loss:.6f}")

    checkpoint = torch.load('./DATA/models/shang_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def train_dip_model(image_shape):
    """Создание Deep Image Prior модели"""
    print("\n" + "="*70)
    print("7. DIP МОДЕЛЬ (2020)")
    print("="*70)

    model = DeepImagePrior(image_shape, latent_dim=100, n_channels=64)

    print("  ✓ DIP модель создана (будет обучаться на каждом тестовом изображении)")
    return model


def train_pgnet_model(system_matrix, image_shape):
    """Обучение модели PGNet (Wu et al., 2023)"""
    print("\n" + "=" * 70)
    print("8. ОБУЧЕНИЕ PGNET МОДЕЛИ (2023)")
    print("=" * 70)

    # Загрузка данных
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4
    output_shape = image_shape

    print(f"  input_dim = {input_dim}")
    print(f"  output_shape = {output_shape}")

    # Используем подвыборку
    n_samples = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    X_train_vec = []
    print("  Подготовка данных...")
    for meas in tqdm(X_train, desc="  Преобразование измерений"):
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part[0], real_part[1], imag_part[0], imag_part[1]])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    # Нормализация
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_vec = scaler.fit_transform(X_train_vec)

    np.save('./DATA/models/pgnet_scaler_mean.npy', scaler.mean_)
    np.save('./DATA/models/pgnet_scaler_scale.npy', scaler.scale_)

    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    model = PGNet(input_dim, output_shape, hidden_dim=256, num_heads=8)
    model = model.to(device)

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("  Начало обучения PGNet модели...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(30), desc="Обучение PGNet")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred_flat = y_pred.view(y_pred.shape[0], -1)
            loss = criterion(y_pred_flat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_
            }, './DATA/models/pgnet_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('PGNet Training Progress')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('./DATA/results/training_curves/pgnet_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  PGNet модель обучена, лучшая loss: {best_loss:.6f}")

    checkpoint = torch.load('./DATA/models/pgnet_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def train_deq_model(system_matrix, image_shape):
    """Обучение модели DEQ-MPI (Güngör et al., 2024)"""
    print("\n" + "="*70)
    print("9. ОБУЧЕНИЕ DEQ-MPI МОДЕЛИ (2024)")
    print("="*70)

    # Загрузка данных
    if os.path.exists('./DATA/dataset/synthetic_complete_X_train.npy'):
        print("  Использование расширенного синтетического датасета...")
        X_train = np.load('./DATA/dataset/synthetic_complete_X_train.npy')
        y_train = np.load('./DATA/dataset/synthetic_complete_y_train.npy')
    else:
        X_train = np.load('./DATA/dataset/X_train.npy')
        y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем подвыборку
    n_samples = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4

    X_train_vec = []
    print("  Подготовка данных...")
    for meas in tqdm(X_train, desc="  Преобразование измерений"):
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)
    y_train_img = y_train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    # Создание модели
    n_iterations = 5
    lambda_param = 0.1

    model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
    model = model.to(device)

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_img[:, np.newaxis, :, :], dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"  Начало обучения DEQ-MPI модели (итераций={n_iterations})...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(30), desc="Обучение DEQ-MPI")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'n_iterations': n_iterations,
                'lambda_param': lambda_param
            }, './DATA/models/deq_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('DEQ-MPI Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/deq_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ DEQ-MPI модель обучена, лучшая loss: {best_loss:.6f}")
    return model


def build_pmcnet_variants(system_matrix, image_shape,
                          n_iterations=1500, n_colors=2,
                          init_tau_seconds=2.0e-6):
    """Создаёт три варианта PMCNet (Huang et al., 2026) для пайплайна.

    Все три data-free (обучения не требуют) и принимают одинаковый формат
    частотно-доменных измерений из калибровки сканера, что позволяет
    выставить их в одну таблицу сравнения.

      1. PMCNet-Standard         — измеренная системная матрица S из
                                   `system_matrix` (как в существующих
                                   методах Tikhonov/Kaczmarz).
      2. PMCNet-Physics-Enhanced — синтетическая S, построенная из
                                   аналитической физики MPI (стабильный
                                   Langevin, радиальная s(r), Лиссажу,
                                   ∂/∂t через фиксированную свёртку).
                                   NN не трогаем.
      3. PMCNet-Final            — синтетическая S + Debye-релаксация
                                   с обучаемой τ_k, multi-color (K=
                                   `n_colors`), TV-регуляризация.

    Args:
        system_matrix: комплексная (M, N) системная матрица из
            BeihangUniversityData/OpenMPI — нужна Standard и для
            определения формы синтетических SM остальных вариантов.
        image_shape: (H, W) реконструируемого изображения.
        n_iterations: число шагов оптимизации на одно измерение
            (в статье 20000; здесь 1500 для скорости пайплайна —
            достаточно для двух-капельных фантомов).
        n_colors: число типов МНЧ для Final (multi-color PMCNet).
        init_tau_seconds: начальное значение τ_k для Final
            (2 мкс по Sec. IV.D статьи).

    Returns:
        (standard, physics_enhanced, final) — три реконструктора.
    """
    print("\n" + "=" * 70)
    print("9.5. ИНИЦИАЛИЗАЦИЯ PMCNet (2026) — ТРИ ВАРИАНТА")
    print("=" * 70)

    base_cfg = PMCNetConfig(
        image_size=tuple(image_shape),
        n_iterations=n_iterations,
        learning_rate=1e-3,
    )
    n_meas_bins = int(system_matrix.shape[0])

    # 1. Standard: измеренная SM
    standard = PMCNetStandard(system_matrix, image_shape, config=base_cfg)
    print(f"  ✓ 1) PMCNet-Standard: измеренная SM ({n_meas_bins}×{image_shape[0]*image_shape[1]}), "
          f"L1, {n_iterations} итер.")

    # 2. Physics-Enhanced: аналитическая SM, NN не трогаем
    physics_enhanced = PMCNetPhysicsEnhanced(image_shape, n_meas_bins, config=base_cfg)
    print(f"  ✓ 2) PMCNet-Physics-Enhanced: аналитическая SM "
          f"(Langevin_safe, радиальная s(r), Лиссажу, FD-conv ∂/∂t)")

    # 3. Final: аналитическая SM + Debye + multi-color + TV
    final_cfg = PMCNetConfig(**{
        **base_cfg.__dict__,
        'n_colors': n_colors,
        'use_debye': True,
        'init_tau_seconds': init_tau_seconds,
        'lambda_tv': 1e-3,
    })
    final = PMCNetFinal(image_shape, n_meas_bins,
                        config=final_cfg, n_colors=n_colors)
    print(f"  ✓ 3) PMCNet-Final: аналитическая SM + Debye(τ_init={init_tau_seconds*1e6:.1f}мкс) "
          f"+ multi-color(K={n_colors}) + TV(λ={final_cfg.lambda_tv})")

    return standard, physics_enhanced, final


# Backwards-compatible alias for older callers
def build_pmcnet_reconstructors(system_matrix, image_shape,
                                n_iterations=1500, n_colors=2,
                                init_tau_seconds=2.0e-6):
    """[legacy] Возвращает (standard, final) — для совместимости со старым кодом.

    Новый код должен использовать `build_pmcnet_variants`, который
    возвращает все три варианта.
    """
    standard, _, final = build_pmcnet_variants(
        system_matrix, image_shape,
        n_iterations=n_iterations,
        n_colors=n_colors,
        init_tau_seconds=init_tau_seconds,
    )
    return standard, final


def plot_all_training_curves():
    """Построение сводного графика всех кривых обучения"""
    print("\n" + "="*70)
    print("10. ВИЗУАЛИЗАЦИЯ КРИВЫХ ОБУЧЕНИЯ")
    print("="*70)

    models = {
        'CNN': './DATA/results/training_curves/cnn_training.png',
        'MoDL': './DATA/results/training_curves/modl_training.png',
        'Diffusion': './DATA/results/training_curves/diffusion_training.png',
        'Chae': './DATA/results/training_curves/chae_training.png',
        'Shang': './DATA/results/training_curves/shang_training.png',
        'PGNet': './DATA/results/training_curves/pgnet_training.png',
        'DEQ-MPI': './DATA/results/training_curves/deq_training.png'
    }

    print("  Кривые обучения сохранены в ./DATA/results/training_curves/")
    for name, path in models.items():
        if os.path.exists(path):
            print(f"    ✓ {name}: {path}")


def run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer,
                        chae_model, shang_model, dip_model, pgnet_model, deq_model,
                        pmcnet_standard=None,
                        pmcnet_physics_enhanced=None,
                        pmcnet_final=None):
    """
    Проверка всех моделей на синтетических данных и данных OpenMPI.

    PMCNet передаётся тремя вариантами одновременно
    (Standard / Physics-Enhanced / Final) — это даёт чистое сравнение
    "что даёт измеренная SM", "что даёт аналитическая SM" и
    "что добавляют NN-оптимизации поверх".
    """
    comparator = MPIReconstructionComparator('./../ChineseData/BeihangUniversityData/SystemMatrix.h5')
    comparator.set_cnn_model(cnn_trainer)
    comparator.set_modl_model(modl_trainer)
    comparator.set_diffusion_model(diffusion_trainer)
    comparator.set_chae_model(chae_model)
    comparator.set_shang_model(shang_model)
    comparator.set_dip_model(dip_model)
    comparator.set_pgnet_model(pgnet_model)
    comparator.set_deq_model(deq_model)
    comparator.set_pmcnet_standard(pmcnet_standard)
    comparator.set_pmcnet_physics_enhanced(pmcnet_physics_enhanced)
    comparator.set_pmcnet_final(pmcnet_final)

    # ===================== ЭТАП 1: СИНТЕТИЧЕСКИЕ ДАННЫЕ =====================
    print("\n" + "=" * 70)
    print("11. ПРОВЕРКА МОДЕЛЕЙ НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 70)

    radius = 0.2
    distances = [0.2, 0.15, 0.1, 0.05, 0.025]

    print(f"\n  Радиус капель: {radius}")
    print(f"  Расстояния между центрами: {distances}")

    synthetic_results = comparator.run_full_comparison(radius=radius, distances=distances)

    comparator.print_summary_table()
    comparator.save_results_to_file()

    # ======================= ЭТАП 2: ДАННЫЕ OPENMPI =======================
    print("\n" + "=" * 70)
    print("12. ПРОВЕРКА МОДЕЛЕЙ НА ДАННЫХ OPENMPI")
    print("=" * 70)

    openmpi_results = validate_models_on_openmpi(comparator)

    return {'synthetic': synthetic_results, 'openmpi': openmpi_results}


def load_cnn_model():
    """Загрузка сохраненной CNN модели"""
    print("\nЗагрузка CNN модели...")

    model_path = './DATA/models/cnn_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        base_filters = 32

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'enc1.0.weight' in state_dict:
                input_channels = state_dict['enc1.0.weight'].shape[1]
            else:
                input_channels = 4

            if 'final_conv.3.weight' in state_dict:
                output_channels = state_dict['final_conv.3.weight'].shape[0]
            else:
                output_channels = 1
        else:
            input_channels = 4
            output_channels = 1

        from .models import MPIReconstructionCNN
        model = MPIReconstructionCNN(
            input_channels=input_channels,
            output_channels=output_channels,
            base_filters=base_filters
        )

        trainer = MPITrainer()
        trainer.model = model
        trainer.model.to(trainer.device)
        trainer.model_type = 'cnn'
        trainer.setup_training(learning_rate=1e-3)

        model_state_dict = checkpoint['model_state_dict']
        trainer.model.load_state_dict(model_state_dict, strict=False)

        print("  ✓ CNN модель загружена")
        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_cnn_trainer(
            input_channels=4, output_channels=1, base_filters=32
        )
        return trainer


def load_modl_model(system_matrix, image_shape):
    """Загрузка сохраненной MoDL модели"""
    print("\nЗагрузка MoDL модели...")

    model_path = './DATA/models/modl_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        n_iterations = checkpoint.get('n_iterations', 3)
        lambda_param = checkpoint.get('lambda_param', 0.01)
        base_filters = checkpoint.get('base_filters', 32)

        trainer = ModelTrainerFactory.create_modl_trainer(
            system_matrix=system_matrix,
            image_shape=image_shape,
            n_iterations=n_iterations,
            lambda_param=lambda_param,
            learning_rate=1e-3,
            base_filters=base_filters
        )

        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ MoDL модель загружена")
        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_modl_trainer(
            system_matrix=system_matrix,
            image_shape=image_shape,
            n_iterations=3,
            lambda_param=0.01,
            base_filters=32
        )
        return trainer


def load_diffusion_model():
    """Загрузка сохраненной диффузионной модели"""
    print("\nЗагрузка Diffusion модели...")

    model_path = './DATA/models/diffusion_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        n_steps = checkpoint.get('n_steps', 500)
        base_filters = checkpoint.get('base_filters', 64)

        trainer = ModelTrainerFactory.create_diffusion_trainer(
            n_steps=n_steps,
            learning_rate=1e-4,
            image_size=51,
            base_filters=base_filters
        )

        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ Diffusion модель загружена")
        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_diffusion_trainer(
            n_steps=500,
            base_filters=64
        )
        return trainer


def load_chae_model(system_matrix, image_shape):
    """Загрузка модели Chae (2017)"""
    print("\nЗагрузка Chae модели...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4
    output_dim = image_shape[0] * image_shape[1]

    model_path = './DATA/models/chae_best.pth'

    if os.path.exists(model_path):
        model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ Chae модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)
        return model


def load_shang_model():
    """Загрузка модели Shang et al. (2020)"""
    print("\nЗагрузка Shang модели...")

    model_path = './DATA/models/shang_best.pth'

    if os.path.exists(model_path):
        model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ Shang модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)
        return model


def load_pgnet_model(system_matrix, image_shape):
    """Загрузка модели PGNet (Wu et al., 2023)"""
    print("\nЗагрузка PGNet модели...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4

    model_path = './DATA/models/pgnet_best.pth'

    if os.path.exists(model_path):
        model = PGNet(input_dim, image_shape, hidden_dim=256, num_heads=8)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ PGNet модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = PGNet(input_dim, image_shape, hidden_dim=256, num_heads=8)
        return model


def load_deq_model(system_matrix, image_shape):
    """Загрузка модели DEQ-MPI (Güngör et al., 2024)"""
    print("\nЗагрузка DEQ-MPI модели...")

    model_path = './DATA/models/deq_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        n_iterations = checkpoint.get('n_iterations', 3)
        lambda_param = checkpoint.get('lambda_param', 0.1)

        model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ DEQ-MPI модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = DEQMPI(system_matrix, image_shape, n_iterations=3, lambda_param=0.1)
        return model


def run_pipeline(num_samples=5000, train_models=True):
    """Запуск полного пайплайна с обучением всех моделей"""
    print("=" * 70)
    print("ПАЙПЛАЙН СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
    print("=" * 70)
    print("\nДоступные методы:")
    print("  1. Тихонов (1963) - классическая регуляризация")
    print("  2. KatsMarc (1937) - алгоритм Качмажа (ART)")
    print("  3. Chae (2017) - однослойная полносвязная сеть")
    print("  4. DIP (2020) - Deep Image Prior")
    print("  5. Shang (2020) - CNN для улучшения разрешения")
    print("  6. PGNet (2023) - Projection Generation Network")
    print("  7. DEQ-MPI (2024) - Deep Equilibrium Model")
    print("  8. PMCNet-Std (2026)   - PMCNet Standard, измеренная SM")
    print("  9. PMCNet-Phys (2026)  - PMCNet с улучшенной физикой (аналитическая SM)")
    print(" 10. PMCNet-Final (2026) - PMCNet физика + NN-оптимизации (Debye+multi-color+TV)")
    print(" 11. CNN - улучшенная UNet")
    print(" 12. MoDL - Model-based Deep Learning")
    print(" 13. Diffusion - диффузионная модель")

    create_directories()

    # Генерация расширенных синтетических данных
    synthetic_generator, synthetic_datasets = generate_synthetic_dataset_advanced(num_samples)

    # Загрузка системной матрицы
    generator = MPIDatasetGenerator()
    system_matrix = generator.SM
    image_shape = generator.image_shape

    print(f"\n  Системная матрица: {system_matrix.shape}")
    print(f"  Размер изображения: {image_shape}")

    # Определяем, нужно ли обучать модели
    if train_models:
        print("\n" + "=" * 35)
        print("РЕЖИМ: ОБУЧЕНИЕ НОВЫХ МОДЕЛЕЙ")
        print("=" * 35)

        cnn_trainer = train_cnn_model()
        modl_trainer = train_modl_model(system_matrix, image_shape)
        diffusion_trainer = train_diffusion_model()
        chae_model = train_chae_model(system_matrix, image_shape)
        shang_model = train_shang_model()
        dip_model = train_dip_model(image_shape)
        pgnet_model = train_pgnet_model(system_matrix, image_shape)
        deq_model = train_deq_model(system_matrix, image_shape)

        plot_all_training_curves()
    else:
        print("\n" + "=" * 35)
        print("РЕЖИМ: ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ")
        print("=" * 35)

        cnn_trainer = load_cnn_model()
        modl_trainer = load_modl_model(system_matrix, image_shape)
        diffusion_trainer = load_diffusion_model()
        chae_model = load_chae_model(system_matrix, image_shape)
        shang_model = load_shang_model()
        dip_model = train_dip_model(image_shape)
        pgnet_model = load_pgnet_model(system_matrix, image_shape)
        deq_model = load_deq_model(system_matrix, image_shape)

    # PMCNet — data-free, без отдельного шага «обучения». Три варианта
    # создаются одинаково в обоих режимах (train/load) и принимают тот
    # же частотно-доменный формат измерений, что и остальные методы.
    pmcnet_standard, pmcnet_physics_enhanced, pmcnet_final = build_pmcnet_variants(
        system_matrix, image_shape, n_iterations=1500, n_colors=2,
    )

    # Проверка моделей
    comparison = run_full_comparison(
        cnn_trainer, modl_trainer, diffusion_trainer,
        chae_model, shang_model, dip_model, pgnet_model, deq_model,
        pmcnet_standard=pmcnet_standard,
        pmcnet_physics_enhanced=pmcnet_physics_enhanced,
        pmcnet_final=pmcnet_final,
    )

    print("\n" + "=" * 70)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    print("=" * 70)
    print("\nРезультаты сохранены в:")
    print("  ./DATA/results/ - изображения сравнения")
    print("  ./DATA/results/all_methods_summary.txt - текстовый отчет")
    print("  ./DATA/results/openmpi_comparison.png - проверка на OpenMPI")
    print("  ./DATA/results/training_curves/ - кривые обучения")
    print("  ./DATA/results/phantoms/ - визуализация фантомов")
    print("  ./DATA/models/ - обученные модели")
    print("  ./DATA/dataset/synthetic_complete_*.npy - синтетический датасет")

    return comparison['synthetic']


def validate_models_on_openmpi(comparator):
    """Валидация всех моделей на датасете OpenMPI"""
    print("\n" + "=" * 70)
    print("ВАЛИДАЦИЯ НА ДАТАСЕТЕ OPENMPI")
    print("=" * 70)

    results = comparator.compare_all_on_openmpi()
    return results


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='MPI Reconstruction Pipeline')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples for dataset generation')
    parser.add_argument('--train_models', action='store_true', default=True,
                       help='Train models from scratch')
    parser.add_argument('--load_models', action='store_true',
                       help='Load pre-trained models')

    args = parser.parse_args()

    train_models = args.train_models
    if args.load_models:
        train_models = False

    results = run_pipeline(num_samples=args.num_samples, train_models=train_models)

    if results:
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ МЕТОДАМ:")
        print("="*70)

        best_ssim = None
        best_method = None

        for result in results:
            distance = result['distance']
            print(f"\n📍 Расстояние: {distance:.3f}")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name:15s}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}дБ, "
                      f"FWHM={m['fwhm']:.2f}px, время={m['time']:.4f}c")

                if best_ssim is None or m['ssim'] > best_ssim:
                    best_ssim = m['ssim']
                    best_method = name

        print("\n" + "="*70)
        print(f"🏆 ЛУЧШИЙ МЕТОД: {best_method} (SSIM = {best_ssim:.4f})")
        print("="*70)


if __name__ == '__main__':
    main()