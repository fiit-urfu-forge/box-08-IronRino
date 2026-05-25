"""Генератор датасета для MPI реконструкции"""

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os


class MPIDatasetGenerator:
    """Генератор датасета для обучения CNN на MPI реконструкции"""

    def __init__(self, system_matrix_path='./../ChineseData/BeihangUniversityData/SystemMatrix.h5'):
        self.load_system_matrix(system_matrix_path)

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
        print("Загрузка системной матрицы...")
        fSM = h5py.File(path, 'r')
        S_data_r = fSM['/measurement/data/r'][:]
        S_data_i = fSM['/measurement/data/i'][:]
        S = S_data_r + 1j * S_data_i
        isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()
        S = S[:, :, isBG == 0]
        self.SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

        number_Position = fSM['/calibration/size'][:].squeeze()
        self.nx, self.ny = int(number_Position[0]), int(number_Position[1])
        self.image_shape = (self.nx, self.ny)

        print(f"Размер SM: {self.SM.shape}")
        print(f"Размер изображения: {self.image_shape}")
        fSM.close()

    def generate_two_droplets(self, radius, distance, intensity1=0.7, intensity2=0.7):
        """Генерация изображения с двумя каплями"""
        image = np.zeros((self.nx, self.ny))

        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        center1_x = -distance / 2
        center2_x = distance / 2

        sigma = radius / 2.5

        r1 = np.sqrt((X - center1_x) ** 2 + Y ** 2)
        image += intensity1 * np.exp(-(r1 ** 2) / (2 * sigma ** 2))

        r2 = np.sqrt((X - center2_x) ** 2 + Y ** 2)
        image += intensity2 * np.exp(-(r2 ** 2) / (2 * sigma ** 2))

        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_random_image(self):
        """Генерация случайного изображения для обучения"""
        num_droplets = np.random.randint(1, 5)

        image = np.zeros((self.nx, self.ny))
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        for _ in range(num_droplets):
            center_x = np.random.uniform(-0.8, 0.8)
            center_y = np.random.uniform(-0.8, 0.8)
            radius = np.random.uniform(0.05, 0.3)
            intensity = np.random.uniform(0.3, 1.0)
            sigma = radius / 2.5

            r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            droplet = intensity * np.exp(-(r ** 2) / (2 * sigma ** 2))

            overlap_threshold = 0.3
            if np.max(image + droplet) <= 1.0 + overlap_threshold:
                image = np.maximum(image, droplet)
            else:
                image = image + 0.5 * droplet

        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_measurement(self, image):
        """Генерация измерений из изображения"""
        image_vector = image.reshape(-1, 1)
        measurement = self.SM @ image_vector
        measurement_reshaped = measurement.reshape(2, -1)
        return measurement_reshaped

    def create_dataset(self, num_samples=5000, test_size=0.2):
        """Создание полного датасета"""
        print(f"Создание датасета из {num_samples} образцов...")

        images = []
        measurements = []

        for i in range(num_samples):
            if i % 500 == 0:
                print(f"  Создано {i}/{num_samples} образцов...")

            if np.random.random() < 0.7:
                image = self.generate_random_image()
            else:
                radius = np.random.uniform(0.05, 0.3)
                distance = np.random.uniform(radius, min(2 * radius, 1.5))
                image = self.generate_two_droplets(radius, distance)

            measurement = self.generate_measurement(image)

            images.append(image)
            measurements.append(measurement)

        images = np.array(images, dtype=np.float32)
        measurements = np.array(measurements, dtype=np.complex64)

        X_train, X_test, y_train, y_test = train_test_split(
            measurements, images, test_size=test_size, random_state=42
        )

        print(f"Размеры датасета:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

        np.save('./DATA/dataset/X_train.npy', X_train)
        np.save('./DATA/dataset/X_test.npy', X_test)
        np.save('./DATA/dataset/y_train.npy', y_train)
        np.save('./DATA/dataset/y_test.npy', y_test)
        np.save('./DATA/dataset/SM.npy', self.SM)
        np.save('./DATA/dataset/image_shape.npy', self.image_shape)

        return X_train, X_test, y_train, y_test


class MPIDataset:
    """PyTorch Dataset для MPI реконструкции - ИСПРАВЛЕННЫЙ"""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)  # Измерения: (samples, 2, n_measurements)
        self.y = np.load(y_path)  # Изображения: (samples, nx, ny)

        print(f"Dataset shapes: X={self.X.shape}, y={self.y.shape}")
        print(f"Image shape from data: {self.y[0].shape}")

        self.n_measurements = self.X.shape[2]

        # Для входных данных CNN нужно 4 канала: real1, imag1, real2, imag2
        # Измерения: 2 катушки * (real + imag) = 4 канала
        self.input_channels = 4

        # Находим размер квадратного изображения для измерений
        # 4 канала * (size * size) должно быть достаточно для размещения всех измерений
        self.measurement_size = self._find_square_size(self.n_measurements)
        print(f"Measurement square size: {self.measurement_size}")

    def _find_square_size(self, n):
        """Находит ближайший квадратный размер"""
        size = int(np.ceil(np.sqrt(n)))
        return size

    def _prepare_measurement(self, measurement):
        """Подготовка измерений для CNN"""
        # measurement: (2, n_measurements) комплексные
        real_part = measurement.real  # (2, n_measurements)
        imag_part = measurement.imag  # (2, n_measurements)

        # Объединяем в 4 канала: real1, imag1, real2, imag2
        # Формируем как [real1, imag1, real2, imag2] -> (4, n_measurements)
        combined = np.zeros((4, self.n_measurements), dtype=np.float32)
        combined[0, :] = real_part[0, :]  # real1
        combined[1, :] = imag_part[0, :]  # imag1
        combined[2, :] = real_part[1, :]  # real2
        combined[3, :] = imag_part[1, :]  # imag2

        # Преобразуем в квадратную форму
        target_size = self.measurement_size * self.measurement_size

        if self.n_measurements < target_size:
            # Дополняем нулями
            padded = np.zeros((4, target_size), dtype=np.float32)
            padded[:, :self.n_measurements] = combined
            combined = padded
        elif self.n_measurements > target_size:
            # Обрезаем если нужно
            combined = combined[:, :target_size]

        # Reshape в квадрат (4, size, size)
        measurement_square = combined.reshape(4, self.measurement_size, self.measurement_size)

        return measurement_square

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        measurement = self.X[idx]
        measurement_prepared = self._prepare_measurement(measurement)

        target = self.y[idx]
        target = np.expand_dims(target, axis=0)  # (1, nx, ny)

        return (torch.tensor(measurement_prepared, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))