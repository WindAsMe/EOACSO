from dataclasses import dataclass, field


@dataclass
class OptimizationResult:
    best_position: object
    best_fitness: float
    best_info: dict
    history: list = field(default_factory=list)  # best fitness per generation
    n_evaluations: int = 0
    archive: list = field(default_factory=list)  # only populated by EOACSO_Paper
