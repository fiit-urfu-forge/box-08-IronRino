"""Подпакет с моделями реконструкции MPI.

Группировка по статьям/назначению:

  classical.py  — TikhonovReconstructor, KatsMarcAlgorithm
  chae.py       — ChaeSingleLayerNN, ChaeMultiLayerNN          (Chae 2017)
  dip.py        — DeepImagePrior                                 (Dittmer 2020)
  shang.py      — ShangCNN (= FDSMPI)                            (Shang 2022)
  deq.py        — DEQMPI, RDNBlock, LCBlock                      (Güngör 2024)
  pmcnet.py     — PMCNet + два уточнения                          (Huang 2026)
  simple.py     — MPIReconstructionCNN, MoDLNetwork, DiffusionModel (baselines)
"""

from .classical import TikhonovReconstructor, KatsMarcAlgorithm
from .chae import ChaeSingleLayerNN, ChaeMultiLayerNN
from .dip import DeepImagePrior
from .shang import ShangCNN, FDSMPI
from .deq import DEQMPI, RDNBlock, LCBlock
from .simple import MPIReconstructionCNN, MoDLNetwork, DiffusionModel
from .moe import MoEReconstructor
from .pmcnet import (
    PMCNetConfig,
    PMCNet,
    PMCNetWithRefinedPhysics,
    PMCNetWithAnalyticalPhysics,
    PMCNetReconstructor,
    PMCNetRefinedReconstructor,
    PMCNetAnalyticalReconstructor,
    PMCNetStandard,
    PMCNetPhysicsEnhanced,
    PMCNetFinal,
    DebyeRelaxationFilter,
    SystemMatrixForward,
    RadialCoilSensitivity,
    LissajousFFPTrajectory,
    TimeDerivativeFD,
    LangevinMagnetization,
    AnalyticalForwardModel,
    build_analytical_system_matrix,
    PMCNetUNet,
    langevin_safe,
)

__all__ = [
    # Classical
    'TikhonovReconstructor', 'KatsMarcAlgorithm',
    # Per-paper
    'ChaeSingleLayerNN', 'ChaeMultiLayerNN',
    'DeepImagePrior',
    'ShangCNN', 'FDSMPI',
    'DEQMPI', 'RDNBlock', 'LCBlock',
    # Baselines
    'MPIReconstructionCNN', 'MoDLNetwork', 'DiffusionModel',
    # Mixture of Experts
    'MoEReconstructor',
    # PMCNet (trio + helpers)
    'PMCNetConfig',
    'PMCNet', 'PMCNetWithRefinedPhysics', 'PMCNetWithAnalyticalPhysics',
    'PMCNetReconstructor', 'PMCNetRefinedReconstructor',
    'PMCNetAnalyticalReconstructor',
    'PMCNetStandard', 'PMCNetPhysicsEnhanced', 'PMCNetFinal',
    'DebyeRelaxationFilter', 'SystemMatrixForward',
    'RadialCoilSensitivity', 'LissajousFFPTrajectory',
    'TimeDerivativeFD', 'LangevinMagnetization',
    'AnalyticalForwardModel', 'build_analytical_system_matrix',
    'PMCNetUNet', 'langevin_safe',
]
