"""Präzisions-, Optimizer- und Scheduler-Module für minotaur_exp."""

from minotaur_exp.optim.muon_8bit import (
    ApolloMiniOptimizer,
    LOMOFusedHookController,
    Muon8BitOptimizer,
    build_optimizer,
)
from minotaur_exp.optim.precision import (
    AdaFP8Linear,
    COATSavedTensorContext,
    convert_model_pure_bf16,
    fp32_to_bf16_sr,
    replace_linear_with_ada_fp8,
)
from minotaur_exp.optim.schedule import (
    ModelEMA,
    Rho1SparseBackwardSchedule,
    WSDScheduler,
)

__all__ = [
    "AdaFP8Linear",
    "ApolloMiniOptimizer",
    "COATSavedTensorContext",
    "LOMOFusedHookController",
    "ModelEMA",
    "Muon8BitOptimizer",
    "Rho1SparseBackwardSchedule",
    "WSDScheduler",
    "build_optimizer",
    "convert_model_pure_bf16",
    "fp32_to_bf16_sr",
    "replace_linear_with_ada_fp8",
]
