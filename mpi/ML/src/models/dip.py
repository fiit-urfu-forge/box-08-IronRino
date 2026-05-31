"""Deep Image Prior (DIP) — реконструкция MPI без обучающих данных.

## Идея Deep Image Prior

Архитектура сверточной сети сама по себе является «приором» (априорным
предпочтением) для естественных изображений. Если оптимизировать веса
случайно инициализированной сети так, чтобы её выход совпадал с
зашумлённым измерением, она НА РАННИХ ИТЕРАЦИЯХ сначала восстанавливает
плавную структуру (которую легко выразить через свёртки), и только потом
«дорастает» до шума. Этот эффект известен как `spectral bias` свёрточных
сетей: low-frequency content учится быстрее, high-frequency — медленнее.

Используя early stopping (остановка ДО переобучения к шуму), мы получаем
de-noised реконструкцию, не имея ни одного примера для обучения.

## Алгоритм DIP для MPI (Dittmer et al., 2020)

    Init θ ~ random, z ~ N(0,1) (фиксированный латент-вход)
    Loop iter = 1 ... N (типично N = 20000):
        ĉ = φ_θ(z)                      # 3D-автоэнкодер генерирует c
        û = S · ĉ                       # система. матрица: c → сигнал
        L = ||û − u_meas||₁             # L1-невязка
        L.backward(); Adam.step()       # обновляем веса сети
    Return ĉ_at_best_iter                # early stopping по PSNR/SSIM

## Отличия от классического DIP

  • **Архитектура — автоэнкодер БЕЗ skip-соединений**. В оригинальном
    DIP (Ulyanov et al. 2018) для inpainting/denoising использовалась
    U-Net с shallow skip. Здесь skip-связи убраны: они пропускают
    высокочастотный шум из энкодера прямо в декодер, минуя `spectral
    bias` боттлнека. Без skip-связей сеть вынуждена кодировать всё
    через узкое горло → low-frequency content предпочтительнее.

  • **ReLU на выходе** (не Sigmoid). Концентрация МНЧ — физически
    неотрицательная величина, но не ограничена сверху. ReLU(out)
    гарантирует c ≥ 0 и оставляет амплитуду свободной.

  • **L1-loss** вместо MSE. На квантифицированном измерительном сигнале
    MPI шум имеет negativeнеnormal распределение (выбросы от АЦП);
    L1 более устойчив к выбросам.

## Сильные и слабые стороны

  ✓ Не требует обучающего датасета (одно измерение → одна реконструкция);
  ✓ Не привязан к конкретному сканеру через калибровочный набор;
  ✓ Конкурентоспособен по качеству с supervised методами на open MPI dataset.

  ✗ Медленный: 5000–20000 итераций оптимизации на каждое измерение;
  ✗ Критичен выбор момента остановки (early stopping). Слишком рано —
    недореконструкция; слишком поздно — переобучение к шуму.

## Сравнение с PMCNetPaper

DIP и PMCNetPaper — оба data-free методы с одинаковой структурой цикла.
Главное различие — в физической модели:
  • DIP использует S·c (системная матрица как линейный оператор);
  • PMCNetPaper использует аналитическую физику через Langevin, Lissajous,
    интеграл — без предварительной калибровки.

Можно считать DIP «предшественником» PMCNet, где SM заменяется явной
физической моделью.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepImagePrior(nn.Module):
    """DIP-генератор: φ_θ(z) → c, обучается на одном измерении.

    Внутренняя архитектура — симметричный 2D-автоэнкодер с 4 уровнями
    downsampling/upsampling. SKIP-СОЕДИНЕНИЯ ОТСУТСТВУЮТ
    (за исключением tied connection «вход энкодера = выход декодера»).
    """

    def __init__(self, image_shape, latent_channels: int = 1,
                 base_channels: int = 32):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.latent_channels = latent_channels

        # Энкодер: 4 уровня
        self.enc1 = self._block(latent_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        # Декодер: 4 уровня; БЕЗ skip-соединений между энкодером и декодером
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=2, stride=2)
        self.dec4 = self._block(base_channels * 4, base_channels * 4)
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=2, stride=2)
        self.dec3 = self._block(base_channels * 2, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=2, stride=2)
        self.dec2 = self._block(base_channels, base_channels)
        out_base = max(base_channels // 2, 8)
        self.up1 = nn.ConvTranspose2d(base_channels, out_base,
                                      kernel_size=2, stride=2)
        self.dec1 = self._block(out_base, out_base)

        # ReLU на выходе — обеспечивает неотрицательность концентрации
        # (статья явно указывает ReLU, не Sigmoid)
        self.head = nn.Sequential(
            nn.Conv2d(out_base, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Подгоняем размер входа под кратное 16 для корректной работы pool
        H, W = self.image_shape
        H_in = ((H + 15) // 16) * 16
        W_in = ((W + 15) // 16) * 16
        if z.shape[-2:] != (H_in, W_in):
            z = F.interpolate(z, size=(H_in, W_in),
                              mode='bilinear', align_corners=False)

        x = self.pool(self.enc1(z))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))

        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        x = self.head(x)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W),
                              mode='bilinear', align_corners=False)
        return x

    def generate_random_latent(self, batch_size: int = 1) -> torch.Tensor:
        """Случайный фиксированный вход z ~ N(0, I) формы (B, C, H, W)."""
        return torch.randn(batch_size, self.latent_channels, *self.image_shape)


__all__ = ['DeepImagePrior']
