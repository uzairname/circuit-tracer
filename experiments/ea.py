from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any
import torch
import numpy as np

from circuit_tracer.graph import Graph


# =============================================================================
# Individual Representation
# =============================================================================

class Individual(ABC):
    """Abstract base class for an individual (solution) in the population.
    
    An individual represents a specific pruning configuration for the graph.
    Subclasses define how the pruning is encoded (e.g., binary masks, thresholds).
    """
    
    @abstractmethod
    def decode(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode this individual into node and edge masks.
        
        Args:
            graph: The graph being pruned
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (node_mask, edge_mask)
                - node_mask: Boolean tensor indicating which nodes to keep
                - edge_mask: Boolean tensor indicating which edges to keep
        """
        pass
    
    @abstractmethod
    def copy(self) -> "Individual":
        """Create a deep copy of this individual."""
        pass


IndividualT = TypeVar("IndividualT", bound=Individual)


# =============================================================================
# Fitness / Objective Functions
# =============================================================================

@dataclass
class FitnessResult:
    """Container for fitness evaluation results.
    
    Attributes:
        objectives: Dict mapping objective names to their values
        constraints: Dict mapping constraint names to their violation values (0 = satisfied)
        metadata: Optional dict for additional info (e.g., node counts, timing)
    """
    objectives: dict[str, float]
    constraints: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied."""
        return all(v <= 0 for v in self.constraints.values())
    
    def dominates(self, other: "FitnessResult", maximize: set[str] | None = None) -> bool:
        """Check if this solution dominates another (Pareto dominance).
        
        Args:
            other: Another fitness result to compare against
            maximize: Set of objective names to maximize (others are minimized)
            
        Returns:
            True if this solution dominates the other
        """
        maximize = maximize or set()
        dominated = False
        for name, val in self.objectives.items():
            other_val = other.objectives[name]
            if name in maximize:
                if val < other_val:
                    return False
                if val > other_val:
                    dominated = True
            else:
                if val > other_val:
                    return False
                if val < other_val:
                    dominated = True
        return dominated


class FitnessFunction(ABC):
    """Abstract base class for fitness/objective evaluation.
    
    Responsible for computing objective values for a given individual.
    """
    
    def __init__(self, graph: Graph):
        """
        Args:
            graph: The raw (unpruned) graph to evaluate against
        """
        self.graph = graph
    
    @abstractmethod
    def evaluate(self, individual: Individual) -> FitnessResult:
        """Evaluate the fitness of an individual.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            FitnessResult containing objective values and any constraint violations
        """
        pass
    
    @property
    @abstractmethod
    def objective_names(self) -> list[str]:
        """Names of the objectives being optimized."""
        pass
    
    @property
    @abstractmethod
    def maximize_objectives(self) -> set[str]:
        """Set of objective names that should be maximized (others minimized)."""
        pass


# =============================================================================
# Genetic Operators
# =============================================================================

class Initializer(ABC, Generic[IndividualT]):
    """Abstract base class for population initialization."""
    
    def __init__(self, graph: Graph):
        """
        Args:
            graph: The graph being optimized
        """
        self.graph = graph
    
    @abstractmethod
    def initialize(self, n: int) -> list[IndividualT]:
        """Generate initial population of n individuals.
        
        Args:
            n: Number of individuals to create
            
        Returns:
            List of initialized individuals
        """
        pass


class Crossover(ABC, Generic[IndividualT]):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, parent1: IndividualT, parent2: IndividualT) -> tuple[IndividualT, IndividualT]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        pass


class Mutation(ABC, Generic[IndividualT]):
    """Abstract base class for mutation operators."""
    
    def __init__(self, graph: Graph):
        """
        Args:
            graph: The graph being optimized (may be used for informed mutations)
        """
        self.graph = graph
    
    @abstractmethod
    def mutate(self, individual: IndividualT) -> IndividualT:
        """Mutate an individual (in-place or return new).
        
        Args:
            individual: The individual to mutate
            
        Returns:
            The mutated individual
        """
        pass


class Selection(ABC, Generic[IndividualT]):
    """Abstract base class for selection operators."""
    
    @abstractmethod
    def select(
        self, 
        population: list[IndividualT], 
        fitness_results: list[FitnessResult],
        n: int
    ) -> list[IndividualT]:
        """Select n individuals from the population.
        
        Args:
            population: Current population
            fitness_results: Fitness results for each individual (same order)
            n: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        pass


class Repair(ABC, Generic[IndividualT]):
    """Abstract base class for repair operators.
    
    Repairs invalid individuals to satisfy constraints.
    """
    
    def __init__(self, graph: Graph):
        """
        Args:
            graph: The graph being optimized
        """
        self.graph = graph
    
    @abstractmethod
    def repair(self, individual: IndividualT) -> IndividualT:
        """Repair an individual to make it valid.
        
        Args:
            individual: The individual to repair
            
        Returns:
            The repaired individual
        """
        pass


# =============================================================================
# Archive / Population Management
# =============================================================================

class Archive(ABC, Generic[IndividualT]):
    """Abstract base class for maintaining an archive of solutions.
    
    Used to track the Pareto front or other elite solutions.
    """
    
    @abstractmethod
    def update(
        self, 
        individuals: list[IndividualT], 
        fitness_results: list[FitnessResult]
    ) -> None:
        """Update the archive with new individuals.
        
        Args:
            individuals: New individuals to consider
            fitness_results: Corresponding fitness results
        """
        pass
    
    @abstractmethod
    def get_front(self) -> list[tuple[IndividualT, FitnessResult]]:
        """Get the current Pareto front (or elite set).
        
        Returns:
            List of (individual, fitness) tuples on the front
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of solutions in the archive."""
        pass


# =============================================================================
# Termination Criteria
# =============================================================================

class TerminationCriterion(ABC):
    """Abstract base class for termination conditions."""
    
    @abstractmethod
    def should_terminate(
        self,
        generation: int,
        population: list[Individual],
        fitness_results: list[FitnessResult],
        archive: Archive | None = None
    ) -> bool:
        """Check if evolution should terminate.
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_results: Fitness of current population
            archive: Optional archive of elite solutions
            
        Returns:
            True if evolution should stop
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the termination criterion state for a new run."""
        pass


# =============================================================================
# Callbacks / Logging
# =============================================================================

class Callback(ABC):
    """Abstract base class for evolution callbacks.
    
    Used for logging, visualization, checkpointing, etc.
    """
    
    def on_generation_start(
        self,
        generation: int,
        population: list[Individual]
    ) -> None:
        """Called at the start of each generation."""
        pass
    
    def on_generation_end(
        self,
        generation: int,
        population: list[Individual],
        fitness_results: list[FitnessResult],
        archive: Archive | None = None
    ) -> None:
        """Called at the end of each generation."""
        pass
    
    def on_evolution_start(self) -> None:
        """Called when evolution begins."""
        pass
    
    def on_evolution_end(
        self,
        final_population: list[Individual],
        final_fitness: list[FitnessResult],
        archive: Archive | None = None
    ) -> None:
        """Called when evolution ends."""
        pass


# =============================================================================
# Main EA Framework
# =============================================================================

@dataclass
class EAConfig:
    """Configuration for the evolutionary algorithm."""
    population_size: int = 100
    max_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    random_seed: int | None = None


class EvolutionaryAlgorithm(ABC, Generic[IndividualT]):
    """Abstract base class for the main evolutionary algorithm.
    
    Orchestrates the evolution process using pluggable components.
    """
    
    def __init__(
        self,
        graph: Graph,
        config: EAConfig,
        fitness_function: FitnessFunction,
        initializer: Initializer[IndividualT],
        crossover: Crossover[IndividualT],
        mutation: Mutation[IndividualT],
        selection: Selection[IndividualT],
        archive: Archive[IndividualT] | None = None,
        repair: Repair[IndividualT] | None = None,
        termination: TerminationCriterion | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        Args:
            graph: The raw graph to optimize pruning for
            config: EA configuration
            fitness_function: Evaluates individuals
            initializer: Creates initial population
            crossover: Crossover operator
            mutation: Mutation operator
            selection: Selection operator
            archive: Optional archive for elite solutions
            repair: Optional repair operator for invalid individuals
            termination: Termination criterion (default: max generations)
            callbacks: List of callbacks for logging/visualization
        """
        self.graph = graph
        self.config = config
        self.fitness_function = fitness_function
        self.initializer = initializer
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.archive = archive
        self.repair = repair
        self.termination = termination
        self.callbacks = callbacks or []
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
    
    @abstractmethod
    def evolve(self) -> tuple[list[IndividualT], list[FitnessResult]]:
        """Run the evolutionary algorithm.
        
        Returns:
            Tuple of (final_population, final_fitness_results)
        """
        pass
    
    def evaluate_population(
        self, 
        population: list[IndividualT]
    ) -> list[FitnessResult]:
        """Evaluate fitness of all individuals in population.
        
        Args:
            population: List of individuals to evaluate
            
        Returns:
            List of fitness results (same order as population)
        """
        return [self.fitness_function.evaluate(ind) for ind in population]
