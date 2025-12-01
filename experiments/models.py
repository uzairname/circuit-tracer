from dataclasses import dataclass, field
from typing import List


@dataclass
class EAHyperparameters:
    """Hyperparameters for NSGA-II multi-objective evolutionary algorithm."""
    
    population_size: int = 50
    n_generations: int = 20
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_sigma: float = 0.1  # Standard deviation for Gaussian mutation
    
    # NSGA-II uses two objectives:
    # Objective 0: Quality (maximize) = completeness + replacement
    # Objective 1: Complexity (minimize) = log(nodes) + log(edges)
    # 
    # Sub-weights within each objective (for combining metrics)
    w_completeness: float = 0.5  # Weight for completeness in quality objective
    w_replacement: float = 1.0   # Weight for replacement in quality objective  
    w_complexity_node: float = 2.0  # Weight for log(nodes) in complexity objective
    w_complexity_edge: float = 1.0  # Weight for log(edges) in complexity objective
    
    # Constraints / Bounds
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    
    # Number of objectives for NSGA-II
    n_objectives: int = 2
