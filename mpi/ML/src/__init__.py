"""MPI Reconstruction Package.

Структура:
  data/      генерация и загрузка данных (phantoms, simulators,
             datasets, openmpi);
  models/    модели реконструкции (classical, chae, dip, shang, deq,
             simple, pmcnet);
  trainer.py инфраструктура обучения;
  comparator.py сравнение методов на синтетике и OpenMPI;
  pipeline.py end-to-end orchestration (entry point — `run_pipeline`).
"""

# Re-exports для удобства внешних пользователей и обратной совместимости
from .data import (
    PhantomType, PhantomGenerator,
    NanoparticleProperties, ChebyshevSystemFunction,
    PhysicalMPISimulator, SyntheticDatasetGenerator,
    MPIDatasetGenerator, MPIDataset,
    OpenMPIDataManager, OpenMPIDataset,
)
from .models import (
    TikhonovReconstructor, KatsMarcAlgorithm,
    ChaeSingleLayerNN, ChaeMultiLayerNN,
    DeepImagePrior, ShangCNN, FDSMPI,
    DEQMPI, RDNBlock, LCBlock,
    MPIReconstructionCNN, MoDLNetwork, DiffusionModel,
    PMCNetConfig,
    PMCNet, PMCNetWithRefinedPhysics, PMCNetWithAnalyticalPhysics,
    PMCNetReconstructor, PMCNetRefinedReconstructor,
    PMCNetAnalyticalReconstructor,
    PMCNetStandard, PMCNetPhysicsEnhanced, PMCNetFinal,
    DebyeRelaxationFilter, SystemMatrixForward,
    RadialCoilSensitivity, LissajousFFPTrajectory,
    TimeDerivativeFD, LangevinMagnetization,
    AnalyticalForwardModel, build_analytical_system_matrix,
    PMCNetUNet, langevin_safe,
)
from .trainer import MPITrainer, ModelTrainerFactory
from .comparator import MPIReconstructionComparator
from .metrics import MetricsCalculator
from .visualization import Visualization
from .pipeline import run_pipeline

__all__ = [
    # Pipeline entry
    'run_pipeline',
    # Data
    'PhantomType', 'PhantomGenerator',
    'NanoparticleProperties', 'ChebyshevSystemFunction',
    'PhysicalMPISimulator', 'SyntheticDatasetGenerator',
    'MPIDatasetGenerator', 'MPIDataset',
    'OpenMPIDataManager', 'OpenMPIDataset',
    # Models
    'TikhonovReconstructor', 'KatsMarcAlgorithm',
    'ChaeSingleLayerNN', 'ChaeMultiLayerNN',
    'DeepImagePrior', 'ShangCNN', 'FDSMPI',
    'DEQMPI', 'RDNBlock', 'LCBlock',
    'MPIReconstructionCNN', 'MoDLNetwork', 'DiffusionModel',
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
    # Training / evaluation
    'MPITrainer', 'ModelTrainerFactory',
    'MPIReconstructionComparator',
    'MetricsCalculator', 'Visualization',
]
