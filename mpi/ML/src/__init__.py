"""MPI Reconstruction Package"""

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .models import (
    MPIReconstructionCNN,
    MoDLNetwork,
    DiffusionModel,
    CombinedHybridModel,
    TikhonovReconstructor,
    ChaeSingleLayerNN,
    DeepImagePrior,
    ShangCNN,
    PGNet,
    DEQMPI,
    KatsMarcAlgorithm,
)
from .pmcnet import (
    PMCNetConfig,
    PMCNet,
    PMCNetWithRefinedPhysics,
    PMCNetWithAnalyticalPhysics,
    PMCNetReconstructor,
    PMCNetRefinedReconstructor,
    PMCNetAnalyticalReconstructor,
    # Три именованных варианта для пайплайна
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
from .trainer import MPITrainer, ModelTrainerFactory
from .comparator import MPIReconstructionComparator
from .metrics import MetricsCalculator
from .visualization import Visualization
from .synthetic_data_generator import (
    SyntheticDatasetGenerator,
    PhysicalMPISimulator,
    PhantomGenerator,
    PhantomType,
    NanoparticleProperties,
    ChebyshevSystemFunction,
    integrate_synthetic_data_to_pipeline,
)

__all__ = [
    # Datasets
    'MPIDatasetGenerator',
    'MPIDataset',
    # Existing models
    'MPIReconstructionCNN',
    'MoDLNetwork',
    'DiffusionModel',
    'CombinedHybridModel',
    'TikhonovReconstructor',
    'ChaeSingleLayerNN',
    'DeepImagePrior',
    'ShangCNN',
    'PGNet',
    'DEQMPI',
    'KatsMarcAlgorithm',
    # PMCNet (Huang et al., 2026) + аналитический физический форвард
    # (LaTeX «Моделирование MPI», Maxwell-PCNN)
    'PMCNetConfig',
    'PMCNet',
    'PMCNetWithRefinedPhysics',
    'PMCNetWithAnalyticalPhysics',
    'PMCNetReconstructor',
    'PMCNetRefinedReconstructor',
    'PMCNetAnalyticalReconstructor',
    # Три именованных варианта для пайплайна:
    'PMCNetStandard',          # измеренная SM
    'PMCNetPhysicsEnhanced',   # аналитическая SM
    'PMCNetFinal',             # аналитическая SM + Debye + multi-color + TV
    'DebyeRelaxationFilter',
    'SystemMatrixForward',
    'RadialCoilSensitivity',
    'LissajousFFPTrajectory',
    'TimeDerivativeFD',
    'LangevinMagnetization',
    'AnalyticalForwardModel',
    'build_analytical_system_matrix',
    'PMCNetUNet',
    'langevin_safe',
    # Training / comparison / utilities
    'MPITrainer',
    'ModelTrainerFactory',
    'MPIReconstructionComparator',
    'MetricsCalculator',
    'Visualization',
    'SyntheticDatasetGenerator',
    'PhysicalMPISimulator',
    'PhantomGenerator',
    'PhantomType',
    'NanoparticleProperties',
    'ChebyshevSystemFunction',
    'integrate_synthetic_data_to_pipeline',
]
