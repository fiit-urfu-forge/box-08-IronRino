"""Подпакет генерации и загрузки данных.

  phantoms.py    — 2D-фантомы (PhantomType, PhantomGenerator).
  simulators.py  — синтетические измерения через физику (Chae 2017) и
                   через системную матрицу (Chebyshev SM).
  datasets.py    — высокоуровневая обёртка над BeihangUniversityData.
  openmpi.py     — загрузка и подготовка OpenMPI-датасета.
"""

from .phantoms import PhantomType, PhantomGenerator
from .simulators import (
    NanoparticleProperties,
    ChebyshevSystemFunction,
    PhysicalMPISimulator,
    SyntheticDatasetGenerator,
)
from .datasets import MPIDatasetGenerator, MPIDataset
from .openmpi import OpenMPIDataManager, OpenMPIDataset

__all__ = [
    'PhantomType', 'PhantomGenerator',
    'NanoparticleProperties', 'ChebyshevSystemFunction',
    'PhysicalMPISimulator', 'SyntheticDatasetGenerator',
    'MPIDatasetGenerator', 'MPIDataset',
    'OpenMPIDataManager', 'OpenMPIDataset',
]
