"""Тренировка нейросетевых моделей с tqdm"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import numpy as np
from tqdm import tqdm


class MPITrainer:
    """Базовый тренер для MPI моделей с tqdm"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"MPITrainer: Используется устройство: {self.device}")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.model_type = None

    def create_model(self, model_type='cnn', **kwargs):
        """Создание модели указанного типа"""
        from .models import MPIReconstructionCNN, MoDLNetwork, DiffusionModel

        self.model_type = model_type

        if model_type == 'cnn':
            self.model = MPIReconstructionCNN(
                input_channels=kwargs.get('input_channels', 4),
                output_channels=kwargs.get('output_channels', 1),
                base_filters=kwargs.get('base_filters', 32),
            )
        elif model_type == 'modl':
            self.model = MoDLNetwork(
                system_matrix=kwargs.get('system_matrix'),
                image_shape=kwargs.get('image_shape'),
                n_iterations=kwargs.get('n_iterations', 3),
                lambda_param=kwargs.get('lambda_param', 0.01),
                base_filters=kwargs.get('base_filters', 32),
            )
        elif model_type == 'diffusion':
            self.model = DiffusionModel(
                n_steps=kwargs.get('n_steps', 100),
                image_size=kwargs.get('image_size', 51),
                base_filters=kwargs.get('base_filters', 64),
                beta_start=kwargs.get('beta_start', 1e-4),
                beta_end=kwargs.get('beta_end', 0.02),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)
        return self.model

    def setup_training(self, learning_rate=1e-3, weight_decay=1e-5):
        """Настройка обучения"""
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),  # AdamW лучше
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  # Более современный
            self.optimizer, T_0=10, T_mult=2
        )

    def train_epoch(self, train_loader):
        """Одна эпоха обучения с tqdm"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for measurements, targets in pbar:
            measurements = measurements.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            model_type_lower = str(self.model_type).lower() if self.model_type else ''

            if 'diffusion' in model_type_lower:
                # DDPM-обучение: модель сама добавляет шум к target и
                # предсказывает его. Условие из measurements здесь не
                # используется (диффузионный baseline безусловный).
                loss = self.model(targets)

            else:
                outputs = self.model(measurements)
                loss = self.criterion(outputs, targets)

            loss.backward()
            # Градиентное клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Валидация с tqdm"""
        self.model.eval()
        total_loss = 0

        pbar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for measurements, targets in pbar:
                measurements = measurements.to(self.device)
                targets = targets.to(self.device)

                model_type_lower = str(self.model_type).lower() if self.model_type else ''

                if 'diffusion' in model_type_lower:
                    loss = self.model(targets)
                else:
                    outputs = self.model(measurements)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, save_path='./DATA/models/mpi_model_best.pth'):
        """Полный цикл обучения с tqdm"""
        print(f"Начало обучения {self.model_type.upper()} модели...")
        print(f"Всего эпох: {epochs}")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        # Создаем общий прогресс-бар для эпох
        epoch_pbar = tqdm(range(epochs), desc='Epochs', position=0)

        for epoch in epoch_pbar:
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_type': self.model_type,
                }, save_path)

            epoch_time = time.time() - start_time

            # Обновляем прогресс-бар
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'best_loss': f'{best_val_loss:.6f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{epoch_time:.1f}s'
            })

        # Загрузка лучшей модели
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_type = checkpoint.get('model_type', self.model_type)
            print(f"\n✓ Загружена лучшая модель из эпохи {checkpoint['epoch'] + 1}")
            print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
            print(f"  Val Loss: {checkpoint['val_loss']:.6f}")

        return train_losses, val_losses

    def predict(self, measurement):
        """Предсказание на новых данных"""
        import numpy as np
        import torch

        self.model.eval()

        with torch.no_grad():
            if isinstance(measurement, np.ndarray):
                if self.model_type == 'cnn':
                    real_part = measurement.real
                    imag_part = measurement.imag
                    combined = np.concatenate([real_part, imag_part], axis=0)

                    n_measurements = combined.shape[1]
                    size = int(np.sqrt(n_measurements))
                    if size * size != n_measurements:
                        size += 1
                        target_size = size * size
                        padded = np.zeros((4, target_size), dtype=np.float32)
                        padded[:, :n_measurements] = combined
                        combined = padded

                    measurement_tensor = combined.reshape(1, 4, size, size)
                    measurement_tensor = torch.tensor(measurement_tensor, dtype=torch.float32)

                elif self.model_type == 'modl':
                    real_part = measurement.real
                    imag_part = measurement.imag
                    meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
                    measurement_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

                elif self.model_type == 'diffusion':
                    # Для диффузионной модели используем обратный процесс
                    real_part = measurement.real
                    imag_part = measurement.imag
                    combined = np.concatenate([real_part, imag_part], axis=0)

                    n_measurements = combined.shape[1]
                    size = int(np.sqrt(n_measurements))
                    if size * size != n_measurements:
                        size += 1
                        target_size = size * size
                        padded = np.zeros((4, target_size), dtype=np.float32)
                        padded[:, :n_measurements] = combined
                        combined = padded

                    condition = combined.reshape(1, 4, size, size)
                    condition = torch.tensor(condition, dtype=torch.float32)
                    condition = F.interpolate(condition, size=(51, 51), mode='bilinear', align_corners=False)

                    # Обратный процесс диффузии
                    batch_size = 1
                    x_t = torch.randn(batch_size, 1, 51, 51).to(self.device)

                    for t in reversed(range(self.model.n_steps)):
                        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                        x_t = self.model.p_sample(x_t, t_tensor, condition)

                    measurement_tensor = x_t

                else:
                    if measurement.ndim == 2 and measurement.shape[0] == 2:
                        real_part = measurement.real
                        imag_part = measurement.imag
                        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
                    else:
                        meas_vector = measurement.flatten()
                    measurement_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

            else:
                measurement_tensor = measurement

            measurement_tensor = measurement_tensor.to(self.device)
            output = self.model(measurement_tensor)

        return output.cpu().numpy()

    def load_model(self, path):
        """Загрузка сохраненной модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_type = checkpoint.get('model_type', self.model_type)
        print(f"✓ Модель загружена из эпохи {checkpoint['epoch'] + 1}")
        print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
        print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
        return checkpoint


class ModelTrainerFactory:
    """Фабрика для создания тренеров разных моделей"""

    @staticmethod
    def create_cnn_trainer(input_channels=4, output_channels=1, learning_rate=1e-3, base_filters=64):
        trainer = MPITrainer()
        trainer.create_model('cnn',
                           input_channels=input_channels,
                           output_channels=output_channels,
                           base_filters=base_filters)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer

    @staticmethod
    def create_modl_trainer(system_matrix, image_shape, n_iterations=5, lambda_param=0.01,
                           learning_rate=1e-3, base_filters=64):
        trainer = MPITrainer()
        trainer.create_model('modl',
                           system_matrix=system_matrix,
                           image_shape=image_shape,
                           n_iterations=n_iterations,
                           lambda_param=lambda_param,
                           base_filters=base_filters)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer

    @staticmethod
    def create_diffusion_trainer(n_steps=100, learning_rate=1e-4, image_size=51,
                                 base_filters=64, beta_start=1e-4, beta_end=0.02):
        trainer = MPITrainer()
        trainer.create_model(
            'diffusion',
            n_steps=n_steps,
            image_size=image_size,
            base_filters=base_filters,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        trainer.setup_training(learning_rate=learning_rate)
        return trainer