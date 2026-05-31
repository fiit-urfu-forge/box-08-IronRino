"""Chae 2017 — реконструкция MPI через прямую регрессию полносвязной сетью.

## Подход

Самый ранний нейросетевой метод реконструкции MPI (ETRI Journal 2017).
Полностью эмпирический: обучает сеть напрямую отображать измеренный
сигнал в спектре гармоник в карту концентрации, без явной физической
модели:

    u (M спектральных бинов) → φ_θ → c (N вокселей)

Сеть учится на парах (u_train, c_train), где u_train генерируется через
известную системную матрицу. После обучения работает за миллисекунды
на новых данных.

## Теоретический результат: связь с Чебышёвым

Анализ обученных весов показал, что столбцы матрицы W сходятся к
**полиномам Чебышёва второго рода** — теоретическому виду обратной
системной функции MPI. То есть сеть фактически переоткрывает
аналитическое решение через данные. Это даёт уверенность, что подход
не «запоминает» обучающие примеры, а выучивает обобщающее
преобразование.

## Реализованы две архитектуры

  ChaeSingleLayerNN  — однослойный перцептрон с сигмоидной активацией
                       y = σ(W·x), без bias-термов. Работает для частиц
                       ≥ 40 нм; проваливается на < 30 нм (нелинейность
                       Ланжевена слишком сильно отклоняется от линейной
                       аппроксимации).

  ChaeMultiLayerNN   — двухслойная сеть с одним скрытым слоем:
                       y = σ(W₂·σ(W₁·x)). Скрытый слой даёт
                       двухпорядковое улучшение MSE; работает для
                       35-нм частиц. Критично: hidden_dim ≥ output_dim
                       (иначе сеть не способна выучить полное обратное
                       отображение).

## Применимость

Размеры обобщены до произвольных input_dim → output_dim:
  • оригинал статьи: 200 (1 катушка × 200 гармоник) → 129 (1D-вокселей);
  • наш 2D MPI: 5100 (2 катушки × 2550 гармоник) → 2601 (51×51 вокселей).

## Ограничения

  • Требует обучающего набора с известной системной матрицей;
  • Не использует физическую модель → нельзя оценить uncertainty;
  • Качество резко падает на частицах с диаметром меньше калибровочного.
"""

import torch
import torch.nn as nn


class ChaeSingleLayerNN(nn.Module):
    """Однослойный перцептрон с сигмоидной активацией (Chae 2017, Sec. II.A).

    Архитектура буквально из статьи: y = σ(W·x), **без bias** (статья:
    "Considering no bias terms"). Анализ W показывает, что её столбцы
    стремятся к полиномам Чебышёва второго рода — это, по сути,
    обучаемое псевдо-обращение системной матрицы.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class ChaeMultiLayerNN(nn.Module):
    """Двухслойная сеть со скрытым слоем (Chae 2017, Sec. II.B).

    Архитектура из статьи: y = σ(W₂ · σ(W₁·x)). Скрытый слой в
    оригинале 200 нейронов (для output_dim=129).

    **Критичный момент из статьи (Sec. III.3):** «The training is
    difficult to achieve for a number of hidden units smaller than the
    length of the target vector». То есть `hidden_dim ≥ output_dim`
    обязательно. Поэтому default — `max(200, ⌈1.5·output_dim⌉)`,
    а не фиксированные 200.

    Bias-термов нет, как в статье.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            # Гарантируем hidden ≥ output (см. doc)
            hidden_dim = max(200, int(output_dim * 1.5))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ['ChaeSingleLayerNN', 'ChaeMultiLayerNN']
