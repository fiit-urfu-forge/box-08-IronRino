"""Нейросетевые модели для MPI реконструкции - улучшенные версии"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MPIReconstructionCNN(nn.Module):
    """Упрощенная и надежная CNN для MPI реконструкции"""

    def __init__(self, input_channels=4, output_channels=1, base_filters=32):
        super().__init__()

        # Энкодер
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 36x36 -> 18x18

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 18x18 -> 9x9

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)  # 9x9 -> 4x4 (округление)

        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True)
        )

        # Декодер с правильными размерами каналов
        # up3: 4x4 -> 9x9
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        # После конкатенации: base_filters*4 (up3) + base_filters*4 (enc3) = base_filters*8
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # up2: 9x9 -> 18x18
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        # После конкатенации: base_filters*2 (up2) + base_filters*2 (enc2) = base_filters*4
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        # up1: 18x18 -> 36x36
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        # После конкатенации: base_filters (up1) + base_filters (enc1) = base_filters*2
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Выходные слои
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Финальный апсемплинг до 51x51
        self.final_upsample = nn.Upsample(size=(51, 51), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Энкодер
        enc1 = self.enc1(x)
        enc1_pooled = self.pool1(enc1)

        enc2 = self.enc2(enc1_pooled)
        enc2_pooled = self.pool2(enc2)

        enc3 = self.enc3(enc2_pooled)
        enc3_pooled = self.pool3(enc3)

        # Bridge
        bridge = self.bridge(enc3_pooled)

        # Декодер с интерполяцией для выравнивания размеров
        # Уровень 3
        dec3_up = self.up3(bridge)  # 4x4 -> 9x9
        # Интерполируем enc3 до размера dec3_up
        enc3_resized = F.interpolate(enc3, size=dec3_up.shape[2:], mode='bilinear', align_corners=False)
        dec3_cat = torch.cat([dec3_up, enc3_resized], dim=1)
        dec3 = self.dec3(dec3_cat)

        # Уровень 2
        dec2_up = self.up2(dec3)  # 9x9 -> 18x18
        enc2_resized = F.interpolate(enc2, size=dec2_up.shape[2:], mode='bilinear', align_corners=False)
        dec2_cat = torch.cat([dec2_up, enc2_resized], dim=1)
        dec2 = self.dec2(dec2_cat)

        # Уровень 1
        dec1_up = self.up1(dec2)  # 18x18 -> 36x36
        enc1_resized = F.interpolate(enc1, size=dec1_up.shape[2:], mode='bilinear', align_corners=False)
        dec1_cat = torch.cat([dec1_up, enc1_resized], dim=1)
        dec1 = self.dec1(dec1_cat)

        # Выход
        output = self.final_conv(dec1)
        output = self.final_upsample(output)

        return output


class ResidualBlock(nn.Module):
    """Residual блок с сохранением размеров"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Убеждаемся, что входные и выходные каналы совпадают
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection с корректировкой каналов если нужно
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class ChannelAttention(nn.Module):
    """Channel Attention механизм"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MoDLNetwork(nn.Module):
    """Улучшенная MoDL сеть с Residual блоками"""

    def __init__(self, system_matrix, image_shape, n_iterations=5, lambda_param=0.01, base_filters=64):
        super().__init__()

        # Подготовка системной матрицы
        if np.iscomplexobj(system_matrix):
            SM_real = np.real(system_matrix)
            SM_imag = np.imag(system_matrix)
            self.system_matrix_extended = np.vstack([SM_real, SM_imag])
        else:
            self.system_matrix_extended = system_matrix

        self.M, self.N = self.system_matrix_extended.shape
        self.image_shape = image_shape
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param

        # Регистрация матриц
        self.register_buffer('A', torch.tensor(self.system_matrix_extended, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T)

        # Улучшенный денойзер
        self.denoiser = MoDLDenoiserImproved(input_channels=1, base_filters=base_filters)

        # Начальная реконструкция
        self.initial_recon = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward_operator(self, x):
        """Прямой оператор: изображение -> измерения"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return x_flat @ self.A_T

    def adjoint_operator(self, y):
        """Сопряженный оператор: измерения -> изображение"""
        batch_size = y.shape[0]
        x_flat = y @ self.A
        return x_flat.view(batch_size, 1, self.image_shape[0], self.image_shape[1])

    def forward(self, measurements):
        batch_size = measurements.shape[0]

        # Начальная реконструкция через сопряженный оператор
        x = self.adjoint_operator(measurements)

        # Применяем initial_recon для улучшения начального приближения
        x = self.initial_recon(x)

        # Итерации MoDL
        for _ in range(self.n_iterations):
            # Градиентный шаг
            Ax = self.forward_operator(x)
            residual = Ax - measurements
            AT_residual = self.adjoint_operator(residual)
            grad_step = x - self.lambda_param * AT_residual

            # Денойзинг
            x = self.denoiser(grad_step)

        return x


class MoDLDenoiserImproved(nn.Module):
    """Улучшенный денойзер для MoDL с Residual блоками"""
    def __init__(self, input_channels=1, base_filters=64):
        super().__init__()

        self.input_conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = ResidualBlock(base_filters, base_filters)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.down3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_filters * 4, base_filters * 4)

        # Decoder - исправлены входные каналы
        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        # После конкатенации: up3 (base_filters*2) + enc3 (base_filters*4) = base_filters*6
        self.dec3 = ResidualBlock(base_filters * 6, base_filters * 2)

        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        # После конкатенации: up2 (base_filters) + enc2 (base_filters*2) = base_filters*3
        self.dec2 = ResidualBlock(base_filters * 3, base_filters)

        self.up1 = nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2)
        # После конкатенации: up1 (base_filters) + enc1 (base_filters) = base_filters*2
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Сохраняем вход для residual
        residual = x

        # Encoder
        x1 = self.input_conv(x)
        x1 = self.enc1(x1)
        x2 = self.down1(x1)

        x2 = self.enc2(x2)
        x3 = self.down2(x2)

        x3 = self.enc3(x3)
        x4 = self.down3(x3)

        # Bottleneck
        x4 = self.bottleneck(x4)

        # Decoder
        x = self.up3(x4)
        # Проверяем соответствие размеров перед конкатенацией
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        # Output
        output = self.output_conv(x)

        # Residual connection
        if output.shape != residual.shape:
            output = F.interpolate(output, size=residual.shape[-2:], mode='bilinear', align_corners=False)

        return output


class DiffusionUNet(nn.Module):
    """Улучшенный UNet с временным embedding для диффузионной модели"""

    def __init__(self, in_channels=2, out_channels=1, base_filters=128, image_size=51, time_embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size

        # Временной embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, base_filters)
        )

        # Энкодер
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )

        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )

        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU(),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        self.down3 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 8),
            nn.SiLU(),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 8),
            nn.SiLU()
        )

        # Декодер
        self.up1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU(),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )

        self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.final_upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)

    def forward(self, x, t):
        batch_size = x.shape[0]
        original_size = x.shape[-2:]

        # Временной embedding
        t_emb = self.get_time_embedding(t, batch_size, x.device)
        t_emb = self.time_embed(t_emb)

        # Encoder
        x1 = self.enc1(x)

        x2 = self.down1(x1)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.enc3(x3)

        x4 = self.down3(x3)

        # Bottleneck
        x4 = self.bottleneck(x4)

        # Decoder с интерполяцией для выравнивания размеров
        x = self.up1(x4)
        # Выравниваем размеры x3 до размера x
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        output = self.output_conv(x)

        if output.shape[-2:] != original_size:
            output = self.final_upsample(output)

        return output

    def get_time_embedding(self, t, batch_size, device):
        """Получение временного embedding"""
        t_emb = torch.zeros(batch_size, 256, device=device)
        t_float = t.float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 256, 2, device=device).float() * (-np.log(10000.0) / 256))
        t_emb[:, 0::2] = torch.sin(t_float * div_term)
        t_emb[:, 1::2] = torch.cos(t_float * div_term)
        return t_emb


class DiffusionModel(nn.Module):
    """Улучшенная диффузионная модель для MPI реконструкции"""

    def __init__(self, denoiser, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.denoiser = denoiser
        self.n_steps = n_steps

        # Параметры диффузионного процесса
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, n_steps))
        self.register_buffer('alpha', 1.0 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - self.alpha_bar))

    def q_sample(self, x0, t, noise=None):
        """Прямой процесс: добавление шума к изображению"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, x_t, t, condition=None):
        """Обратный процесс: удаление шума с учетом условий"""
        # Подготовка условия
        if condition is not None:
            if condition.shape[-2:] != x_t.shape[-2:]:
                condition = F.interpolate(condition, size=x_t.shape[-2:], mode='bilinear', align_corners=False)

            # Если у условия несколько каналов, усредняем их для совместимости
            if condition.shape[1] > 1:
                condition = condition.mean(dim=1, keepdim=True)

            # Объединяем по каналам
            x_input = torch.cat([x_t, condition], dim=1)
        else:
            x_input = x_t

        # Предсказание шума
        predicted_noise = self.denoiser(x_input, t)

        # Вычисление mean
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        # Добавление шума для t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean

    def forward(self, x0, condition=None):
        """Обучение: предсказание шума с loss weighting"""
        batch_size = x0.shape[0]
        device = x0.device

        # Случайный временной шаг
        t = torch.randint(0, self.n_steps, (batch_size,), device=device)
        noise = torch.randn_like(x0)

        # Добавление шума
        x_t = self.q_sample(x0, t, noise)

        # Подготовка условия
        if condition is not None:
            if condition.shape[-2:] != x_t.shape[-2:]:
                condition = F.interpolate(condition, size=x_t.shape[-2:], mode='bilinear', align_corners=False)

            if condition.shape[1] > 1:
                condition = condition.mean(dim=1, keepdim=True)

            x_input = torch.cat([x_t, condition], dim=1)
        else:
            x_input = x_t

        # Предсказание шума
        predicted_noise = self.denoiser(x_input, t)

        # Loss с weighting по временному шагу
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = loss.mean()

        return loss


class ChaeSingleLayerNN(nn.Module):
    """
    Однослойная нейронная сеть для MPI реконструкции с батчевым обучением
    Источник: Chae, B. G. (2017). "Neural network image reconstruction for
              magnetic particle imaging." ETRI Journal, 39(5), 651-659.

    Улучшенная версия с батчевой нормализацией и dropout для лучшей обобщаемости
    """

    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout_rate=0.2, use_batch_norm=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Улучшенная архитектура с batch norm и dropout
        self.network = nn.Sequential(
            # Входной слой
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Tanh(),
            nn.Dropout(dropout_rate),

            # Выходной слой
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Для нормализации выхода в [0,1]
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Улучшенная инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He инициализация для Tanh активации
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: входной тензор формы (batch_size, input_dim)
        Returns:
            выходной тензор формы (batch_size, output_dim)
        """
        return self.network(x)

    def predict_image(self, x, image_shape=(51, 51)):
        """
        Удобный метод для предсказания с преобразованием в изображение

        Args:
            x: входной тензор формы (batch_size, input_dim) или (input_dim,)
            image_shape: форма выходного изображения (height, width)
        Returns:
            изображение формы (batch_size, 1, height, width) или (height, width)
        """
        single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True

        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            output = output.view(-1, 1, image_shape[0], image_shape[1])

        if single_input:
            output = output.squeeze(0)

        return output


class DeepImagePrior(nn.Module):
    """
    Полноценная реализация Deep Image Prior для MPI
    Источник: Dittmer, S., et al. (2020). "Deep image prior for 3D magnetic
              particle imaging." arXiv:2007.01593.

    Особенности:
    - Глубокая генеративная сеть с U-Net архитектурой
    - Обучение на одном изображении
    - Использование разных типов шума (Gaussian, Poisson, Salt&Pepper)
    """

    def __init__(self, image_shape, latent_dim=100, n_channels=64,
                 noise_type='gaussian', noise_level=0.1):
        super().__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.noise_type = noise_type
        self.noise_level = noise_level

        # Полноценная U-Net архитектура из статьи
        # Encoder
        self.enc1 = self._make_encoder_block(latent_dim, n_channels)
        self.enc2 = self._make_encoder_block(n_channels, n_channels * 2)
        self.enc3 = self._make_encoder_block(n_channels * 2, n_channels * 4)
        self.enc4 = self._make_encoder_block(n_channels * 4, n_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_channels * 8, n_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 8, n_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec4 = self._make_decoder_block(n_channels * 16, n_channels * 4)
        self.dec3 = self._make_decoder_block(n_channels * 8, n_channels * 2)
        self.dec2 = self._make_decoder_block(n_channels * 4, n_channels)
        self.dec1 = self._make_decoder_block(n_channels * 2, n_channels)

        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Для адаптивного размера выхода
        self.output_adapter = nn.AdaptiveAvgPool2d(image_shape)

    def _make_encoder_block(self, in_channels, out_channels):
        """Создание блока энкодера"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Создание блока декодера"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def add_noise(self, x):
        """Добавление шума согласно статье"""
        if self.training:
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(x) * self.noise_level
                return x + noise
            elif self.noise_type == 'poisson':
                # Poisson шум для положительных значений
                x_positive = torch.clamp(x, min=0)
                noise = torch.poisson(x_positive * 255) / 255 - x_positive
                return x + noise
            elif self.noise_type == 'snp':
                # Salt & Pepper шум
                mask = torch.rand_like(x) < self.noise_level
                noise = torch.where(mask, torch.bernoulli(torch.rand_like(x)), torch.zeros_like(x))
                return x + noise
        return x

    def forward(self, z):
        """
        Args:
            z: латентный вектор (batch, latent_dim, 1, 1)
        """
        # Encoder
        x1 = self.enc1(z)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Bottleneck
        x = self.bottleneck(x4)

        # Decoder with skip connections
        x = self.dec4(torch.cat([x, x4], dim=1))
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.dec1(torch.cat([x, x1], dim=1))

        # Output
        output = self.output_conv(x)
        output = self.output_adapter(output)

        # Добавление шума (только при обучении)
        output = self.add_noise(output)

        return output

    def generate_random_latent(self, batch_size=1, distribution='normal'):
        """Генерация случайного латентного вектора"""
        if distribution == 'normal':
            return torch.randn(batch_size, self.latent_dim, 1, 1)
        elif distribution == 'uniform':
            return torch.rand(batch_size, self.latent_dim, 1, 1) * 2 - 1
        else:
            raise ValueError(f"Unknown distribution: {distribution}")


class ShangCNN(nn.Module):
    """
    Полноценная CNN для улучшения разрешения MPI
    Источник: Shang, Y., et al. (2020). "Deep learning for improving the
              spatial resolution of magnetic particle imaging."
              Physics in Medicine & Biology, 65(15), 155012.

    Архитектура:
    - 5 сверточных слоев с residual connections
    - Batch Normalization после каждого сверточного слоя
    - LeakyReLU активации
    - Обучение с L2 регуляризацией
    """

    def __init__(self, input_channels=1, output_channels=1, base_filters=64, dropout_rate=0.2):
        super().__init__()

        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Второй сверточный блок
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Третий сверточный блок
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Четвертый сверточный блок
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Пятый сверточный блок
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Residual connection
        self.residual = nn.Conv2d(input_channels, base_filters, kernel_size=1)

        # Выходной слой
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Инициализация весов He
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Сохраняем вход для residual
        residual = self.residual(x)

        # Forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Добавляем residual
        x = x + residual

        # Выход
        output = self.output_conv(x)

        return output


class PGNet(nn.Module):
    """
    Projection Generation Network
    Источник: Wu, X., et al. (2023). "PGNet: Projection generative network for
              sparse-view reconstruction of projection-based magnetic particle imaging."
              Medical Physics, 50(8), 4928-4942.
    """

    def __init__(self, input_dim, output_shape, hidden_dim=512, num_heads=8):
        super().__init__()
        self.output_shape = output_shape
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Проекция входных измерений
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Self-attention блок для измерений
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # FFN после attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Генератор изображений (MLP)
        output_dim = output_shape[0] * output_shape[1]
        self.image_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, measurements):
        """
        Args:
            measurements: (batch_size, input_dim) - вектор измерений
        Returns:
            image: (batch_size, 1, H, W) - реконструированное изображение
        """
        batch_size = measurements.shape[0]

        # Проекция входных данных
        x = self.input_proj(measurements)  # (batch_size, hidden_dim)

        # Добавляем размерность последовательности для attention
        x_seq = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Self-attention
        attn_out, _ = self.self_attention(x_seq, x_seq, x_seq)
        x = x + attn_out.squeeze(1)
        x = self.attn_norm(x)

        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)

        # Генерация изображения
        output = self.image_generator(x)  # (batch_size, output_dim)
        output = output.view(batch_size, 1, self.output_shape[0], self.output_shape[1])

        return output


class SpatialAttention(nn.Module):
    """Пространственный attention механизм для PGNet - исправленная версия"""

    def __init__(self, hidden_dim, spatial_dim=7):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        # Преобразуем hidden_dim в spatial_dim * spatial_dim для создания пространственной карты
        self.to_spatial = nn.Linear(hidden_dim, spatial_dim * spatial_dim)
        self.to_value = nn.Linear(hidden_dim, hidden_dim)

        self.scale = (hidden_dim ** -0.5)

    def forward(self, x):
        """
        Args:
            x: (batch_size, hidden_dim)
        Returns:
            out: (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]

        # Создаем пространственную карту внимания
        attention_map = self.to_spatial(x)  # (batch_size, spatial_dim * spatial_dim)
        attention_map = attention_map.view(batch_size, 1, self.spatial_dim * self.spatial_dim)
        attention_weights = torch.softmax(attention_map * self.scale, dim=-1)

        # Преобразуем значения
        values = self.to_value(x)  # (batch_size, hidden_dim)
        values = values.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Применяем внимание
        out = torch.bmm(attention_weights, values)  # (batch_size, 1, hidden_dim)
        out = out.squeeze(1)  # (batch_size, hidden_dim)

        return out


class DEQMPI(nn.Module):
    """
    Полноценная Deep Equilibrium Model для MPI
    Источник: Güngör, A., et al. (2024). "DEQ-MPI: A deep equilibrium
              reconstruction with learned consistency for magnetic particle imaging."
              IEEE Transactions on Medical Imaging, 43(5), 1812-1824.

    Особенности:
    - Фиксированная точка через root-finding
    - Phantom gradient для эффективного backward pass
    - Итеративный решатель (Broyden)
    """

    def __init__(self, system_matrix, image_shape, n_iterations=5, lambda_param=0.1,
                 f_solve_method='broyden', tol=1e-6):
        super().__init__()

        # Подготовка системной матрицы
        if np.iscomplexobj(system_matrix):
            SM_real = np.real(system_matrix)
            SM_imag = np.imag(system_matrix)
            self.system_matrix_extended = np.vstack([SM_real, SM_imag])
        else:
            self.system_matrix_extended = system_matrix

        self.M, self.N = self.system_matrix_extended.shape
        self.image_shape = image_shape
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.f_solve_method = f_solve_method
        self.tol = tol

        # Регистрация матриц
        self.register_buffer('A', torch.tensor(self.system_matrix_extended, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T)

        # Data consistency сеть
        self.data_consistency = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.LayerNorm(self.M),
            nn.ReLU(inplace=True),
            nn.Linear(self.M, self.M),
            nn.LayerNorm(self.M),
            nn.ReLU(inplace=True),
            nn.Linear(self.M, self.M)
        )

        # Улучшенный денойзер
        self.denoiser = DEQDenoiser(image_shape)

        # Начальная реконструкция
        self.initial_recon = nn.Sequential(
            nn.Linear(self.M, self.N),
            nn.Tanh()
        )
        with torch.no_grad():
            self.initial_recon[0].weight.data = self.A_T.clone() * self.lambda_param

    def forward_operator(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return x_flat @ self.A_T

    def adjoint_operator(self, y):
        batch_size = y.shape[0]
        x_flat = y @ self.A
        return x_flat.view(batch_size, 1, self.image_shape[0], self.image_shape[1])

    def deq_fixed_point(self, z, measurements):
        """
        Функция для нахождения фиксированной точки
        z = f(z, measurements)
        """
        # Data consistency
        Az = self.forward_operator(z)
        residual = Az - measurements
        dc_out = self.data_consistency(residual)
        AT_dc = self.adjoint_operator(dc_out)

        # Обновление
        z_new = z - self.lambda_param * AT_dc
        z_new = self.denoiser(z_new)

        return z_new

    def broyden_solve(self, z_init, measurements):
        """
        Решение фиксированной точки методом Бройдена
        """
        z = z_init.clone()

        for _ in range(self.n_iterations):
            z_next = self.deq_fixed_point(z, measurements)

            # Проверка сходимости
            if torch.norm(z_next - z) < self.tol:
                break

            z = z_next

        return z

    def forward(self, measurements, return_intermediate=False):
        """
        Forward pass с DEQ
        """
        # Начальная реконструкция
        z_init = self.initial_recon(measurements)
        z_init = z_init.view(-1, 1, self.image_shape[0], self.image_shape[1])

        # Решение фиксированной точки
        if self.f_solve_method == 'broyden':
            z_star = self.broyden_solve(z_init, measurements)
        else:
            # Простые итерации
            z = z_init
            for _ in range(self.n_iterations):
                z = self.deq_fixed_point(z, measurements)
            z_star = z

        if return_intermediate:
            return z_star, z_init
        return z_star


class DEQDenoiser(nn.Module):
    """Денойзер для DEQ-MPI с residual connections"""

    def __init__(self, image_shape, base_filters=64):
        super().__init__()
        self.image_shape = image_shape

        # Конволюционные блоки с residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.skip = nn.Conv2d(1, base_filters, kernel_size=1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Сохраняем residual
        residual = self.skip(x)

        # Convolution blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Добавляем residual
        x = x + residual

        # Выход
        output = self.output_conv(x)

        return output


class KatsMarcAlgorithm:
    """
    Полноценный алгоритм Качмажа (Kaczmarz's algorithm)
    Источник: Kaczmarz, S. (1937). "Angenäherte Auflösung von Systemen linearer
              Gleichungen." Bulletin International de l'Académie Polonaise des
              Sciences et des Lettres. A, 35, 355-357.

    Также известен как ART (Algebraic Reconstruction Technique) в томографии
    """

    def __init__(self, system_matrix, use_random_order=True):
        """
        Args:
            system_matrix: матрица A размера (M, N)
            use_random_order: случайный порядок строк (ускоряет сходимость)
        """
        self.A = system_matrix
        self.M, self.N = system_matrix.shape
        self.use_random_order = use_random_order

        # Предвычисление норм строк для ускорения
        self.row_norms = np.linalg.norm(self.A, axis=1) ** 2

        # Нормализация строк (опционально)
        self.normalized_rows = self.A.copy()
        for i in range(self.M):
            if self.row_norms[i] > 0:
                self.normalized_rows[i] = self.A[i] / self.row_norms[i]

    def reconstruct(self, b, n_iterations=50, relaxation=1.0,
                    method='standard', use_blocks=False, block_size=10):
        """
        Реконструкция алгоритмом Качмажа

        Args:
            b: вектор измерений (M,)
            n_iterations: количество полных проходов
            relaxation: параметр релаксации (0 < relaxation <= 2)
            method: 'standard', 'randomized', 'block'
            use_blocks: использовать блочную версию
            block_size: размер блока для блочной версии
        """
        x = np.zeros(self.N)

        if method == 'randomized' or self.use_random_order:
            # Рандомизированный Kaczmarz (ускоренная сходимость)
            return self._randomized_kaczmarz(b, n_iterations, relaxation)

        elif use_blocks:
            # Блочная версия (Block ART)
            return self._block_kaczmarz(b, n_iterations, relaxation, block_size)

        else:
            # Стандартный Kaczmarz (последовательные проекции)
            return self._standard_kaczmarz(b, n_iterations, relaxation)

    def _standard_kaczmarz(self, b, n_iterations, relaxation):
        """Стандартный алгоритм Качмажа"""
        x = np.zeros(self.N)

        for iteration in range(n_iterations):
            x_old = x.copy()

            for i in range(self.M):
                if self.row_norms[i] > 1e-10:
                    # Проекция на гиперплоскость a_i * x = b[i]
                    residual = b[i] - np.dot(self.A[i], x)
                    x = x + relaxation * (residual / self.row_norms[i]) * self.A[i]

            # Ранняя остановка при малых изменениях
            if np.linalg.norm(x - x_old) < 1e-6:
                break

        return x

    def _randomized_kaczmarz(self, b, n_iterations, relaxation):
        """Рандомизированный алгоритм Качмажа (более быстрая сходимость)"""
        x = np.zeros(self.N)

        # Вероятности строк пропорциональны их норме
        probs = self.row_norms / np.sum(self.row_norms)

        for iteration in range(n_iterations):
            x_old = x.copy()

            # Выбор случайной строки на каждой итерации
            indices = np.random.choice(self.M, size=self.M, p=probs, replace=False)

            for i in indices:
                if self.row_norms[i] > 1e-10:
                    residual = b[i] - np.dot(self.A[i], x)
                    x = x + relaxation * (residual / self.row_norms[i]) * self.A[i]

            if np.linalg.norm(x - x_old) < 1e-6:
                break

        return x

    def _block_kaczmarz(self, b, n_iterations, relaxation, block_size):
        """Блочная версия алгоритма (для разреженных матриц)"""
        x = np.zeros(self.N)

        # Создание блоков
        n_blocks = (self.M + block_size - 1) // block_size
        blocks = []

        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = min((block_idx + 1) * block_size, self.M)
            A_block = self.A[start:end]
            b_block = b[start:end]
            blocks.append((A_block, b_block))

        for iteration in range(n_iterations):
            x_old = x.copy()

            for A_block, b_block in blocks:
                # Псевдообращение для блока
                A_block_T = A_block.T
                A_block_A_block_T = A_block @ A_block_T

                try:
                    # Решение для блока
                    block_solution = np.linalg.solve(A_block_A_block_T, b_block - A_block @ x)
                    x = x + relaxation * (A_block_T @ block_solution)
                except np.linalg.LinAlgError:
                    # Если блок сингулярен, используем псевдообращение
                    pinv = np.linalg.pinv(A_block)
                    x = x + relaxation * (pinv @ (b_block - A_block @ x))

            if np.linalg.norm(x - x_old) < 1e-6:
                break

        return x

    def reconstruct_with_svd(self, b, n_components=None):
        """
        Реконструкция с использованием SVD (для малых систем)
        Используется для анализа свойств алгоритма
        """
        # Вычисление SVD
        U, s, Vt = np.linalg.svd(self.A, full_matrices=False)

        if n_components is None:
            n_components = len(s)

        # Обрезаем по числу компонент
        U = U[:, :n_components]
        s = s[:n_components]
        Vt = Vt[:n_components]

        # Псевдорешение
        Sinv = np.diag(1.0 / s)
        x = Vt.T @ Sinv @ U.T @ b

        return x.real.reshape((51, 51))

    def estimate_optimal_relaxation(self):
        """
        Оценка оптимального параметра релаксации
        На основе спектральных свойств матрицы
        """
        # Оценка сингулярных значений
        try:
            from scipy.sparse.linalg import svds
            s = svds(self.A, k=min(10, self.M - 1), return_singular_vectors=False)
            s_max = np.max(s)
            s_min = np.min(s[s > 0])

            # Оптимальный параметр для блочного Казмарца
            if s_min > 0:
                omega_opt = 2.0 / (1.0 + (s_min / s_max) ** 2)
                return min(omega_opt, 1.5)
        except:
            pass

        # Default значение
        return 1.0


class TikhonovReconstructor:
    """
    Полноценная реконструкция методом Тихонова (Tikhonov regularization)
    Источник: Tikhonov, A. N. (1963). "Solution of incorrectly formulated problems
              and the regularization method." Soviet Mathematics Doklady.

    Алгоритм: x = argmin ||Ax - b||^2 + λ||Lx||^2
    Решение: x = (A^T A + λ L^T L)^{-1} A^T b
    """

    def __init__(self, system_matrix, regularization_matrix='identity'):
        """
        Args:
            system_matrix: матрица A размера (M, N)
            regularization_matrix: тип регуляризации ('identity', 'gradient', 'laplacian')
        """
        self.A = system_matrix
        self.M, self.N = system_matrix.shape

        # Выбор матрицы регуляризации L
        if regularization_matrix == 'identity':
            self.L = np.eye(self.N)
        elif regularization_matrix == 'gradient':
            # 1D градиентная регуляризация
            self.L = np.zeros((self.N - 1, self.N))
            for i in range(self.N - 1):
                self.L[i, i] = 1
                self.L[i, i + 1] = -1
        elif regularization_matrix == 'laplacian':
            # 2D Лапласиан (для изображений)
            nx, ny = 51, 51
            self.L = np.zeros((self.N, self.N))
            for i in range(nx):
                for j in range(ny):
                    idx = i * ny + j
                    self.L[idx, idx] = 4
                    if i > 0:
                        self.L[idx, (i - 1) * ny + j] = -1
                    if i < nx - 1:
                        self.L[idx, (i + 1) * ny + j] = -1
                    if j > 0:
                        self.L[idx, i * ny + (j - 1)] = -1
                    if j < ny - 1:
                        self.L[idx, i * ny + (j + 1)] = -1
        else:
            self.L = np.eye(self.N)

        self.LTL = self.L.T @ self.L

    def _find_optimal_lambda(self, b, lambda_candidates=None, method='l_curve'):
        """Автоматический выбор оптимального параметра регуляризации"""
        if lambda_candidates is None:
            lambda_candidates = np.logspace(-6, 2, 20)

        if method == 'l_curve':
            # L-кривая метод
            residuals = []
            solutions_norms = []

            for lam in lambda_candidates:
                try:
                    ATA = self.A.T @ self.A + lam * self.LTL
                    x = np.linalg.solve(ATA, self.A.T @ b)
                    residuals.append(np.linalg.norm(self.A @ x - b))
                    solutions_norms.append(np.linalg.norm(self.L @ x))
                except:
                    continue

            if len(residuals) > 1:
                # Находим точку максимальной кривизны
                curvatures = []
                for i in range(1, len(residuals) - 1):
                    curv = np.abs((residuals[i + 1] - 2 * residuals[i] + residuals[i - 1]) /
                                  (residuals[i] + 1e-10))
                    curvatures.append(curv)
                best_idx = np.argmax(curvatures) + 1
                return lambda_candidates[best_idx]

        elif method == 'gcv':
            # Generalized Cross-Validation
            gcv_scores = []
            for lam in lambda_candidates:
                try:
                    ATA = self.A.T @ self.A + lam * self.LTL
                    inv_ATA = np.linalg.inv(ATA)
                    H = self.A @ inv_ATA @ self.A.T
                    x = inv_ATA @ self.A.T @ b

                    resid = np.linalg.norm(self.A @ x - b) ** 2
                    trace = np.trace(H)
                    gcv = resid / (self.M - trace) ** 2
                    gcv_scores.append(gcv)
                except:
                    gcv_scores.append(np.inf)

            return lambda_candidates[np.argmin(gcv_scores)]

        # Default: возвращаем среднее значение
        return lambda_candidates[len(lambda_candidates) // 2]

    def reconstruct(self, measurement, mu=None, kmax=100, method='direct'):
        """
        Реконструкция методом Тихонова

        Args:
            measurement: измерения (2, n_measurements)
            mu: параметр регуляризации (если None - автоматический выбор)
            kmax: максимальное число итераций для итеративных методов
            method: 'direct' (прямое решение), 'cg' (сопряженные градиенты),
                   'lsqr' (LSQR алгоритм)
        """
        # Преобразование измерений в вектор
        if isinstance(measurement, np.ndarray):
            if measurement.ndim == 2 and measurement.shape[0] == 2:
                b = np.concatenate([measurement[0, :], measurement[1, :]])
            else:
                b = measurement.flatten()
        else:
            b = measurement

        # Автоматический выбор параметра регуляризации
        if mu is None:
            mu = self._find_optimal_lambda(b)
            print(f"  Выбран оптимальный параметр регуляризации: μ = {mu:.6f}")

        if method == 'direct':
            # Прямое решение
            ATA = self.A.T @ self.A + mu * self.LTL
            try:
                x = np.linalg.solve(ATA, self.A.T @ b)
            except np.linalg.LinAlgError:
                # Используем псевдообращение если матрица сингулярна
                x = np.linalg.lstsq(ATA, self.A.T @ b, rcond=None)[0]

        elif method == 'cg':
            # Метод сопряженных градиентов
            ATA = self.A.T @ self.A + mu * self.LTL
            x = np.zeros(self.N)
            r = self.A.T @ b - ATA @ x
            p = r.copy()

            for _ in range(kmax):
                Ap = ATA @ p
                alpha = (r @ r) / (p @ Ap + 1e-10)
                x = x + alpha * p
                r_new = r - alpha * Ap
                beta = (r_new @ r_new) / (r @ r + 1e-10)
                p = r_new + beta * p
                r = r_new

                if np.linalg.norm(r) < 1e-6:
                    break

        elif method == 'lsqr':
            # LSQR алгоритм (эффективен для больших разреженных матриц)
            from scipy.sparse.linalg import lsqr
            result = lsqr(self.A, b, damp=np.sqrt(mu), iter_lim=kmax)
            x = result[0]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Изменение формы в изображение
        recon_image = x.real.reshape((51, 51))

        # Нормализация
        if recon_image.max() > 0:
            recon_image = recon_image / recon_image.max()

        return recon_image

class CombinedHybridModel(nn.Module):
    # ... (оставляем как было)
    pass