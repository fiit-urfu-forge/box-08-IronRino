"""Подпакет с моделями реконструкции MPI.

Группировка по статьям/назначению:

  classical.py  — TikhonovReconstructor, KatsMarcAlgorithm
  chae.py       — ChaeSingleLayerNN, ChaeMultiLayerNN          (Chae 2017)
  dip.py        — DeepImagePrior                                 (Dittmer 2020)
  pmcnet.py     — PMCNet + три варианта улучшений                 (Huang 2026)
  simple.py     — MPIReconstructionCNN, MoDLNetwork, DiffusionModel (baselines)
"""

from .classical import TikhonovReconstructor, KatsMarcAlgorithm
from .chae import ChaeSingleLayerNN, ChaeMultiLayerNN
from .dip import DeepImagePrior
from .simple import MPIReconstructionCNN, MoDLNetwork, DiffusionModel
from .moe import MoEReconstructor
from .pmcnet import (
    PMCNetConfig,
    PMCNet,
    PMCNetEnhanced,
    PMCNetWithBasicPhysics,
    PMCNetReconstructor,
    PMCNetEnhancedReconstructor,
    PMCNetPaperReconstructor,
    PMCNetStandard,
    PMCNetStdRadialCoil,
    PMCNetStdDebye,
    PMCNetStdSoft,
    PMCNetStdFreqWeighted,
    PMCNetAll,
    PMCNetPaper,
    DebyeRelaxationFilter,
    SystemMatrixForward,
    # Paper-faithful primitives
    UniformCoilSensitivity,
    TimeDerivativeForwardFD,
    BasicAnalyticalForwardModel,
    BasicHardConstrainedSpectralForward,
    # Enhanced primitives
    RadialCoilSensitivity,
    LissajousFFPTrajectory,
    TimeDerivativeFD,
    LangevinMagnetization,
    AnalyticalForwardModel,
    HardConstrainedSpectralForward,
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
    # Baselines
    'MPIReconstructionCNN', 'MoDLNetwork', 'DiffusionModel',
    # Mixture of Experts
    'MoEReconstructor',
    # PMCNet (quartet + helpers)
    'PMCNetConfig',
    'PMCNet', 'PMCNetEnhanced', 'PMCNetWithBasicPhysics',
    'PMCNetReconstructor', 'PMCNetEnhancedReconstructor',
    'PMCNetPaperReconstructor',
    'PMCNetStandard',
    'PMCNetStdRadialCoil', 'PMCNetStdDebye',
    'PMCNetStdSoft', 'PMCNetStdFreqWeighted',
    'PMCNetAll',
    'PMCNetPaper',
    'DebyeRelaxationFilter', 'SystemMatrixForward',
    # Paper-faithful primitives
    'UniformCoilSensitivity', 'TimeDerivativeForwardFD',
    'BasicAnalyticalForwardModel', 'BasicHardConstrainedSpectralForward',
    # Enhanced primitives
    'RadialCoilSensitivity', 'LissajousFFPTrajectory',
    'TimeDerivativeFD', 'LangevinMagnetization',
    'AnalyticalForwardModel', 'HardConstrainedSpectralForward',
    'build_analytical_system_matrix',
    'PMCNetUNet', 'langevin_safe',
]
