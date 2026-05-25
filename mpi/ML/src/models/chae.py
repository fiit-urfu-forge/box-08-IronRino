"""Chae (2017): нейросетевая реконструкция MPI через полносвязные сети.

Источник: Chae B. G. «Neural network image reconstruction for magnetic
particle imaging», ETRI Journal 39(5):651–659, 2017.

Согласно статье, MPI-сигнал — это спектр гармоник (200 на катушку), и
обратная задача отображает их в вектор концентраций (129 вокселей для 1D).
Анализ обученных весов показал, что столбцы матрицы сходятся к полиномам
Чебышёва второго рода — теоретическому виду обратной системной функции.

Реализованы две архитектуры:

  ChaeSingleLayerNN  — однослойный перцептрон с сигмоидной активацией.
                       Работает для частиц ≥ 40 нм, проваливается на < 30 нм.

  ChaeMultiLayerNN   — двухслойная сеть с одним скрытым слоем (200 нейронов).
                       Двухпорядковое улучшение MSE; работает для 35-нм частиц.

В нашем пайплайне размеры обобщены до произвольных input_dim → output_dim
(в оригинале 200 → 129; у нас, например, 5100 → 2601 для 2D MPI).
"""

import torch
import torch.nn as nn


class ChaeSingleLayerNN(nn.Module):
    """Однослойный перцептрон с сигмоидной активацией (Chae 2017, Sec. II.A).

    Архитектура буквально из статьи: y = σ(W·x + b), без скрытых слоёв.
    Анализ W показывает, что её столбцы стремятся к полиномам Чебышёва
    второго рода — это, по сути, обучаемое псевдо-обращение системной
    матрицы.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class ChaeMultiLayerNN(nn.Module):
    """Двухслойная сеть со скрытым слоем (Chae 2017, Sec. II.B).

    Архитектура из статьи: y = σ(W₂ · σ(W₁·x + b₁) + b₂).
    Скрытый слой — 200 нейронов в оригинале; в общем случае задаётся
    параметром `hidden_dim`. Добавление нелинейного слоя усиливает
    классификационные свойства сети и снижает MSE на два порядка.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ['ChaeSingleLayerNN', 'ChaeMultiLayerNN']
