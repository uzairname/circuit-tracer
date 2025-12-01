

from dataclasses import dataclass


@dataclass
class EAHyperparameters:
    population_size: int = 50
    n_generations: int = 20
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_sigma: float = 0.1  # Standard deviation for Gaussian mutation
    
    # Fitness weights
    w_completeness: float = 1.0
    w_replacement: float = 1.0
    w_complexity_node: float = 0.3  # Penalty for log(nodes)
    w_complexity_edge: float = 0.15  # Penalty for log(edges)
    
    # Constraints / Bounds
    min_threshold: float = 0.0
    max_threshold: float = 1.0
