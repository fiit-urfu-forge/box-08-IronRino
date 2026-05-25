"""Загрузчик OpenMPI датасета из локальной папки ChineseData/OpenMPIData.

Данные НЕ скачиваются из сети — используются только локальные .mdf файлы.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Путь к данным OpenMPI относительно каталога ChineseData/
DEFAULT_OPENMPI_SUBPATH = os.path.join('ChineseData', 'OpenMPIData')


def resolve_openmpi_dir():
    """
    Определение пути к локальной папке OpenMPIData.

    Проверяет несколько типовых расположений: текущую директорию,
    родительскую (пайплайн запускается из mpi/ML/) и путь относительно
    самого модуля. Возвращает первый существующий каталог.
    """
    candidates = [
        DEFAULT_OPENMPI_SUBPATH,
        os.path.join('..', DEFAULT_OPENMPI_SUBPATH),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', DEFAULT_OPENMPI_SUBPATH),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.normpath(path)
    # каталог не найден — возвращаем путь относительно модуля
    return os.path.normpath(candidates[-1])


class OpenMPIDataLoader:
    """
    Загрузчик OpenMPI датасета из локальной папки.

    Данные читаются только из локального каталога (по умолчанию
    ChineseData/OpenMPIData) и НЕ скачиваются из сети.
    Источник датасета: https://github.com/MagneticParticleImaging/OpenMPIData.jl

    Ожидаемая структура каталога:
    - calibrations/   — калибровочные данные (системные функции), *.mdf
    - measurements/   — измерения для фантомов, отдельная подпапка на фантом
    """

    # Фантомы по умолчанию (если автоопределение по подпапкам не сработало)
    DEFAULT_PHANTOMS = ['shapePhantom', 'resolutionPhantom',
                        'concentrationPhantom', 'rotationPhantom']

    # Размеры изображений из OpenMPI
    IMAGE_SIZES = {
        '2d': (32, 32),
        '3d': (32, 32, 32)
    }

    def __init__(self, data_dir=None):
        """
        Инициализация загрузчика OpenMPI данных.

        Args:
            data_dir: путь к локальной папке OpenMPIData. Если None —
                      определяется автоматически (ChineseData/OpenMPIData).
        """
        self.data_dir = data_dir if data_dir is not None else resolve_openmpi_dir()
        self.calibrations_dir = os.path.join(self.data_dir, 'calibrations')
        self.measurements_dir = os.path.join(self.data_dir, 'measurements')

        if os.path.isdir(self.data_dir):
            print(f"  OpenMPI данные: {self.data_dir}")
        else:
            print(f"  ВНИМАНИЕ: папка OpenMPI не найдена: {self.data_dir}")
            print("  Поместите данные OpenMPI в эту папку "
                  "(подпапки calibrations/ и measurements/).")

        # Автоопределение доступных фантомов по подпапкам measurements/
        self.PHANTOMS = self._discover_phantoms()

        # Кэш для загруженных данных
        self._calibration_cache = {}
        self._measurement_cache = {}

    def _discover_phantoms(self):
        """Поиск доступных фантомов среди подпапок measurements/"""
        if os.path.isdir(self.measurements_dir):
            phantoms = sorted(
                name for name in os.listdir(self.measurements_dir)
                if os.path.isdir(os.path.join(self.measurements_dir, name))
            )
            if phantoms:
                return phantoms
        return list(self.DEFAULT_PHANTOMS)

    @staticmethod
    def _find_first_file(directory):
        """Первый .mdf файл в каталоге (с числовой сортировкой имён)"""
        if not directory or not os.path.isdir(directory):
            return None

        def sort_key(filename):
            stem = os.path.splitext(filename)[0]
            return (0, int(stem)) if stem.isdigit() else (1, filename)

        files = sorted((f for f in os.listdir(directory)
                        if f.lower().endswith('.mdf')), key=sort_key)
        return os.path.join(directory, files[0]) if files else None

    def load_mdf_file(self, filepath):
        """
        Загрузка локального MDF/HDF5 файла OpenMPI.

        Формат MDF (Magnetic Particle Imaging Data Format) основан на HDF5.
        Возвращает комплексный массив данных или None, если файл
        отсутствует, повреждён или усечён (неполная копия).
        """
        if not filepath or not os.path.exists(filepath):
            print(f"  Файл не найден: {filepath}")
            return None

        try:
            with h5py.File(filepath, 'r') as hf:
                data = None

                # Вариант 1: данные измерений
                if 'measurement' in hf and 'data' in hf['measurement']:
                    data = self._read_complex(hf['measurement/data'])
                # Вариант 2: калибровочные данные
                elif 'calibration' in hf and 'data' in hf['calibration']:
                    data = self._read_complex(hf['calibration/data'])
                # Вариант 3: данные в корне файла
                elif 'data' in hf:
                    data = self._read_complex(hf['data'])

                if data is None:
                    print(f"  ВНИМАНИЕ: не удалось найти данные в {filepath}")
                return data

        except OSError as e:
            print(f"  ВНИМАНИЕ: не удалось прочитать {filepath}")
            print(f"           ({e})")
            print("           Файл повреждён или представляет собой "
                  "неполную (усечённую) копию.")
            return None
        except Exception as e:
            print(f"  Ошибка загрузки {filepath}: {e}")
            return None

    @staticmethod
    def _read_complex(node):
        """
        Извлечение комплексного массива из HDF5 узла.

        Поддерживает три представления комплексных данных:
        - группа с подмассивами 'r' (действ.) и 'i' (мнимая части);
        - массив с native-комплексным типом;
        - действительный массив с последней осью размера 2 (re, im).
        """
        if isinstance(node, h5py.Group):
            if 'r' in node and 'i' in node:
                return node['r'][:] + 1j * node['i'][:]
            return None

        arr = node[:]
        if np.iscomplexobj(arr):
            return arr
        if arr.ndim >= 1 and arr.shape[-1] == 2:
            return arr[..., 0] + 1j * arr[..., 1]
        return arr.astype(np.complex64)

    def load_calibration(self, calibration_type='2d'):
        """
        Загрузка калибровочных данных (системной функции) из calibrations/.

        Args:
            calibration_type: '2d' или '3d'

        Returns:
            system_matrix: системная матрица (n_measurements, n_pixels)
            image_shape: форма изображения
        """
        if calibration_type in self._calibration_cache:
            return self._calibration_cache[calibration_type]

        image_shape = self.IMAGE_SIZES.get(calibration_type, (32, 32))
        calib_path = self._find_first_file(self.calibrations_dir)

        data = self.load_mdf_file(calib_path) if calib_path else None

        if data is not None and data.ndim >= 2:
            # Многомерные данные приводим к (n_measurements, n_pixels)
            if data.ndim >= 3:
                system_matrix = data.reshape(-1, data.shape[-1])
            else:
                system_matrix = data
        else:
            if calib_path is None:
                print("  Калибровочные файлы OpenMPI не найдены в "
                      f"{self.calibrations_dir}")
            print("  Используется тестовая (случайная) системная матрица.")
            n_pixels = int(np.prod(image_shape))
            n_measurements = 2 * n_pixels
            system_matrix = (np.random.randn(n_measurements, n_pixels) +
                             1j * np.random.randn(n_measurements, n_pixels))

        self._calibration_cache[calibration_type] = (system_matrix, image_shape)
        return system_matrix, image_shape

    def load_measurement(self, phantom_name, calibration_type='2d'):
        """
        Загрузка измерений для фантома из measurements/<phantom_name>/.

        Args:
            phantom_name: имя фантома (имя подпапки в measurements/)
            calibration_type: '2d' или '3d'

        Returns:
            measurement: комплексный массив измерений или None
        """
        cache_key = f"{phantom_name}_{calibration_type}"
        if cache_key in self._measurement_cache:
            return self._measurement_cache[cache_key]

        phantom_dir = os.path.join(self.measurements_dir, phantom_name)
        meas_path = self._find_first_file(phantom_dir)

        if meas_path is None:
            print(f"  Измерения для фантома '{phantom_name}' не найдены "
                  f"в {phantom_dir}")
            data = None
        else:
            data = self.load_mdf_file(meas_path)

        self._measurement_cache[cache_key] = data
        return data

    def get_system_matrix_for_2d(self):
        """Получение 2D системной матрицы в удобном формате"""
        SM, image_shape = self.load_calibration('2d')
        return SM, image_shape


class OpenMPIDataset(Dataset):
    """
    PyTorch Dataset для OpenMPI данных
    Создает пары (измерение, изображение) для обучения и тестирования
    """

    def __init__(self, measurements, ground_truths, image_shape=(32, 32), transform=None):
        """
        Args:
            measurements: список или массив измерений
            ground_truths: список или массив эталонных изображений
            image_shape: форма выходного изображения
            transform: трансформации для данных
        """
        self.measurements = measurements
        self.ground_truths = ground_truths
        self.image_shape = image_shape
        self.transform = transform

    def __len__(self):
        return len(self.measurements)

    def _prepare_measurement(self, measurement):
        """Подготовка измерений для нейросети"""
        # measurement shape: (n_frequencies, n_coils, ...)
        if isinstance(measurement, np.ndarray):
            # Преобразование в 4 канала: real1, imag1, real2, imag2
            if measurement.ndim == 3:
                n_freq, n_coils, n_other = measurement.shape
                real_parts = measurement.real  # (n_freq, n_coils, n_other)
                imag_parts = measurement.imag  # (n_freq, n_coils, n_other)

                # Объединение по каналам
                combined = np.concatenate([real_parts, imag_parts], axis=1)  # (n_freq, 2*n_coils, n_other)

                # Изменение формы в квадратное изображение
                side = int(np.ceil(np.sqrt(n_freq)))
                target_size = side * side

                if n_freq < target_size:
                    padded = np.zeros((target_size, combined.shape[1], combined.shape[2]), dtype=np.float32)
                    padded[:n_freq] = combined
                    combined = padded

                # Reshape в (channels, side, side)
                combined = combined.reshape(side, side, combined.shape[1], combined.shape[2])
                combined = combined.transpose(2, 0, 1, 3)  # (channels, side, side, frames)

                # Берем первый кадр
                combined = combined[:, :, :, 0]  # (channels, side, side)

            elif measurement.ndim == 2:
                # (features, frames)
                n_features, n_frames = measurement.shape
                side = int(np.ceil(np.sqrt(n_features)))
                target_size = side * side

                if n_features < target_size:
                    padded = np.zeros((target_size, n_frames), dtype=np.float32)
                    padded[:n_features] = measurement
                    measurement_padded = padded
                else:
                    measurement_padded = measurement[:target_size]

                # Reshape в квадрат
                combined = measurement_padded.reshape(side, side, n_frames)
                combined = combined.transpose(2, 0, 1)  # (frames, side, side)
                combined = combined[0]  # берем первый кадр
                combined = np.expand_dims(combined, axis=0)  # (1, side, side)
            else:
                combined = measurement
        else:
            combined = measurement

        # Изменение размера до целевого
        from scipy.ndimage import zoom
        if combined.shape[-2:] != self.image_shape:
            zoom_factors = (1, self.image_shape[0] / combined.shape[-2],
                            self.image_shape[1] / combined.shape[-1])
            combined = zoom(combined, zoom_factors, order=1)

        return torch.tensor(combined, dtype=torch.float32)

    def __getitem__(self, idx):
        measurement = self.measurements[idx]
        ground_truth = self.ground_truths[idx]

        measurement_tensor = self._prepare_measurement(measurement)

        # Подготовка ground truth
        if isinstance(ground_truth, np.ndarray):
            if ground_truth.ndim == 2:
                ground_truth = np.expand_dims(ground_truth, axis=0)
            elif ground_truth.ndim == 3 and ground_truth.shape[0] != 1:
                ground_truth = ground_truth[0:1]
        else:
            ground_truth = np.array([[ground_truth]])

        if ground_truth.shape[-2:] != self.image_shape:
            from scipy.ndimage import zoom
            zoom_factors = (1, self.image_shape[0] / ground_truth.shape[-2],
                            self.image_shape[1] / ground_truth.shape[-1])
            ground_truth = zoom(ground_truth, zoom_factors, order=1)

        ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)

        if self.transform:
            measurement_tensor = self.transform(measurement_tensor)
            ground_truth_tensor = self.transform(ground_truth_tensor)

        return measurement_tensor, ground_truth_tensor


class OpenMPIDataManager:
    """
    Менеджер для работы с OpenMPI датасетом
    Обеспечивает непересекающиеся и однородные выборки для обучения и тестирования
    """

    def __init__(self, data_dir=None, validation_split=0.2, random_seed=42):
        """
        Args:
            data_dir: путь к локальной папке OpenMPIData (None — автоопределение)
            validation_split: доля данных для тестирования
            random_seed: seed для воспроизводимости разделения
        """
        self.loader = OpenMPIDataLoader(data_dir)
        self.validation_split = validation_split
        self.random_seed = random_seed
        self._data_cache = {}

    def load_and_prepare_data(self, use_synthetic_ground_truth=True):
        """
        Загрузка и подготовка данных для обучения и тестирования

        Args:
            use_synthetic_ground_truth: использовать синтетические ground truth изображения
                                       (на основе положения фантомов)
        """
        print("\n" + "=" * 70)
        print("ЗАГРУЗКА OPENMPI ДАТАСЕТА")
        print("=" * 70)

        # Загрузка системной матрицы
        SM, image_shape = self.loader.get_system_matrix_for_2d()
        print(f"  Системная матрица: {SM.shape}")
        print(f"  Размер изображения: {image_shape}")

        # Сбор всех измерений
        all_measurements = []
        all_ground_truths = []
        all_metadata = []

        for phantom in self.loader.PHANTOMS:
            print(f"\n  Загрузка фантома: {phantom}")

            try:
                measurements = self.loader.load_measurement(phantom, '2d')
                print(f"    Измерения: {measurements.shape if measurements is not None else 'None'}")

                if measurements is not None:
                    # Создание ground truth на основе данных фантома
                    if use_synthetic_ground_truth:
                        ground_truth = self._create_ground_truth_for_phantom(phantom, image_shape)
                    else:
                        ground_truth = self._extract_ground_truth_from_measurements(measurements, image_shape)

                    # Разделение по кадрам
                    n_frames = measurements.shape[-1] if measurements.ndim >= 3 else 1

                    for frame in range(min(n_frames, 20)):  # Ограничиваем количество кадров
                        if measurements.ndim == 3:
                            meas_frame = measurements[:, :, frame]
                        elif measurements.ndim == 2:
                            meas_frame = measurements
                        else:
                            meas_frame = measurements

                        all_measurements.append(meas_frame)
                        all_ground_truths.append(ground_truth)
                        all_metadata.append({'phantom': phantom, 'frame': frame})

                print(f"    Добавлено {len(all_measurements)} образцов")

            except Exception as e:
                print(f"    Ошибка загрузки {phantom}: {e}")
                continue

        # Если реальных данных недостаточно, создаем синтетические
        if len(all_measurements) < 10:
            print("\n  Недостаточно реальных данных, создаем синтетические...")
            synthetic_data = self._generate_synthetic_data(image_shape)
            all_measurements.extend(synthetic_data['measurements'])
            all_ground_truths.extend(synthetic_data['ground_truths'])
            all_metadata.extend(synthetic_data['metadata'])

        print(f"\n  Всего загружено образцов: {len(all_measurements)}")

        # Разделение на обучающую и тестовую выборки (непересекающиеся)
        train_idx, test_idx = self._split_by_phantom(all_metadata, self.validation_split)

        train_measurements = [all_measurements[i] for i in train_idx]
        train_ground_truths = [all_ground_truths[i] for i in train_idx]
        test_measurements = [all_measurements[i] for i in test_idx]
        test_ground_truths = [all_ground_truths[i] for i in test_idx]

        print(f"\n  Обучающая выборка: {len(train_measurements)} образцов")
        print(f"  Тестовая выборка: {len(test_measurements)} образцов")

        # Создание Datasets
        train_dataset = OpenMPIDataset(train_measurements, train_ground_truths, image_shape)
        test_dataset = OpenMPIDataset(test_measurements, test_ground_truths, image_shape)

        # Проверка на непересекающиеся фантомы
        train_phantoms = set([all_metadata[i]['phantom'] for i in train_idx])
        test_phantoms = set([all_metadata[i]['phantom'] for i in test_idx])
        print(f"\n  Фантомы в обучении: {train_phantoms}")
        print(f"  Фантомы в тестировании: {test_phantoms}")
        assert len(train_phantoms.intersection(test_phantoms)) == 0, "ОШИБКА: фантомы пересекаются!"

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'system_matrix': SM,
            'image_shape': image_shape,
            'n_train': len(train_measurements),
            'n_test': len(test_measurements),
            'train_phantoms': train_phantoms,
            'test_phantoms': test_phantoms
        }

    def _split_by_phantom(self, metadata, test_size):
        """
        Разделение данных по фантомам (непересекающиеся выборки)

        Args:
            metadata: список метаданных для каждого образца
            test_size: доля фантомов для тестирования

        Returns:
            train_indices, test_indices
        """
        import random
        random.seed(self.random_seed)

        # Получаем уникальные фантомы
        phantoms = list(set([m['phantom'] for m in metadata]))
        print(f"\n  Уникальные фантомы: {phantoms}")

        # Разделяем фантомы
        n_test_phantoms = max(1, int(len(phantoms) * test_size))
        test_phantoms = set(random.sample(phantoms, n_test_phantoms))
        train_phantoms = set(phantoms) - test_phantoms

        print(f"  Фантомы для тестирования: {test_phantoms}")
        print(f"  Фантомы для обучения: {train_phantoms}")

        # Создаем индексы
        train_indices = [i for i, m in enumerate(metadata) if m['phantom'] in train_phantoms]
        test_indices = [i for i, m in enumerate(metadata) if m['phantom'] in test_phantoms]

        return train_indices, test_indices

    def _create_ground_truth_for_phantom(self, phantom_name, image_shape):
        """
        Создание эталонного изображения на основе имени фантома
        Согласно OpenMPI документации:
        - shapePhantom: коническая форма
        - resolutionPhantom: трубки для оценки разрешения
        - concentrationPhantom: фантом с разной концентрацией
        """
        nx, ny = image_shape

        if phantom_name == 'shapePhantom':
            # Коническая форма
            image = self._create_cone_phantom(nx, ny)
        elif phantom_name == 'resolutionPhantom':
            # Трубки для оценки разрешения
            image = self._create_resolution_phantom(nx, ny)
        elif phantom_name == 'concentrationPhantom':
            # Фантом с разной концентрацией
            image = self._create_concentration_phantom(nx, ny)
        else:
            # Случайный фантом
            image = self._create_random_phantom(nx, ny)

        return image

    def _create_cone_phantom(self, nx, ny):
        """Создание конического фантома (shapePhantom)"""
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Конус с центром в (0,0)
        R = np.sqrt(X ** 2 + Y ** 2)
        cone = np.maximum(0, 1 - R) ** 2

        # Добавляем небольшой градиент
        cone = cone / cone.max() if cone.max() > 0 else cone

        return cone

    def _create_resolution_phantom(self, nx, ny):
        """
        Создание фантома для оценки разрешения
        Несколько трубок на разных расстояниях
        """
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        image = np.zeros((nx, ny))

        # Трубки на разных позициях
        positions = [(-0.6, 0), (-0.2, 0), (0.2, 0), (0.6, 0)]
        radii = [0.08, 0.08, 0.08, 0.08]
        intensities = [0.8, 0.9, 0.9, 0.8]

        for (cx, cy), r, intensity in zip(positions, radii, intensities):
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            tube = intensity * np.exp(-(dist ** 2) / (2 * (r / 2.5) ** 2))
            image = np.maximum(image, tube)

        # Нормализация
        if image.max() > 0:
            image = image / image.max()

        return image

    def _create_concentration_phantom(self, nx, ny):
        """
        Создание фантома с разной концентрацией
        Три области с разной интенсивностью
        """
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        image = np.zeros((nx, ny))

        # Центральная область (высокая концентрация)
        center = np.exp(-((X) ** 2 + (Y) ** 2) / (2 * 0.15 ** 2)) * 1.0

        # Кольцевая область (средняя концентрация)
        R = np.sqrt(X ** 2 + Y ** 2)
        ring = np.exp(-((R - 0.5) ** 2) / (2 * 0.1 ** 2)) * 0.6

        # Угловые области (низкая концентрация)
        corners = np.zeros((nx, ny))
        corner_positions = [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]
        for cx, cy in corner_positions:
            corner = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.12 ** 2)) * 0.3
            corners = np.maximum(corners, corner)

        image = np.maximum(center, ring)
        image = np.maximum(image, corners)

        if image.max() > 0:
            image = image / image.max()

        return image

    def _create_random_phantom(self, nx, ny):
        """Создание случайного фантома"""
        image = np.zeros((nx, ny))

        n_droplets = np.random.randint(1, 5)
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        for _ in range(n_droplets):
            cx = np.random.uniform(-0.7, 0.7)
            cy = np.random.uniform(-0.7, 0.7)
            r = np.random.uniform(0.1, 0.25)
            intensity = np.random.uniform(0.3, 0.9)

            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            droplet = intensity * np.exp(-(dist ** 2) / (2 * (r / 2.5) ** 2))
            image = np.maximum(image, droplet)

        if image.max() > 0:
            image = image / image.max()

        return image

    def _extract_ground_truth_from_measurements(self, measurements, image_shape):
        """
        Извлечение ground truth из измерений (если возможно)
        Использует псевдообращение системной матрицы
        """
        SM, _ = self.loader.get_system_matrix_for_2d()

        try:
            # Преобразование измерений в вектор
            if measurements.ndim == 3:
                n_freq, n_coils, n_frames = measurements.shape
                meas_vector = np.concatenate([measurements[:, :, 0].real.flatten(),
                                              measurements[:, :, 0].imag.flatten()])
            elif measurements.ndim == 2:
                meas_vector = np.concatenate([measurements.real.flatten(),
                                              measurements.imag.flatten()])
            else:
                meas_vector = measurements.flatten()

            # Псевдообращение
            SM_pinv = np.linalg.pinv(SM)
            recon = (SM_pinv @ meas_vector).real
            recon = recon.reshape(image_shape)

            # Нормализация
            if recon.max() > 0:
                recon = recon / recon.max()

            return recon

        except Exception as e:
            print(f"    Ошибка извлечения ground truth: {e}")
            return np.zeros(image_shape)

    def _generate_synthetic_data(self, image_shape):
        """Генерация синтетических данных для расширения датасета"""
        nx, ny = image_shape

        measurements_list = []
        ground_truths_list = []
        metadata_list = []

        # Получаем системную матрицу для симуляции измерений
        SM, _ = self.loader.get_system_matrix_for_2d()

        # Генерируем различные фантомы
        phantom_names = ['synthetic_cone', 'synthetic_resolution', 'synthetic_concentration']

        for phantom in phantom_names:
            # Создаем ground truth
            if 'cone' in phantom:
                gt = self._create_cone_phantom(nx, ny)
            elif 'resolution' in phantom:
                gt = self._create_resolution_phantom(nx, ny)
            else:
                gt = self._create_concentration_phantom(nx, ny)

            # Симулируем измерения
            gt_vector = gt.reshape(-1, 1)
            measurement = (SM @ gt_vector).reshape(2, -1)  # 2 катушки
            measurement = measurement.reshape(SM.shape[0] // 2, 2, -1)

            measurements_list.append(measurement)
            ground_truths_list.append(gt)
            metadata_list.append({'phantom': phantom, 'frame': 0, 'synthetic': True})

        return {
            'measurements': measurements_list,
            'ground_truths': ground_truths_list,
            'metadata': metadata_list
        }

    def create_data_loaders(self, batch_size=8, num_workers=0):
        """
        Создание DataLoader для обучения и тестирования

        Args:
            batch_size: размер батча
            num_workers: количество workers для загрузки данных
        """
        data = self.load_and_prepare_data()

        train_loader = DataLoader(
            data['train_dataset'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )

        test_loader = DataLoader(
            data['test_dataset'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'system_matrix': data['system_matrix'],
            'image_shape': data['image_shape'],
            'n_train': data['n_train'],
            'n_test': data['n_test']
        }
