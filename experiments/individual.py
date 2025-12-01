import torch
from typing import Set, Tuple, List, Optional

class Individual:
    """
    Individual for NSGA-II multi-objective optimization.
    
    Represents a pruning configuration with per-layer thresholds and
    optional node/edge overrides.
    """
    
    def __init__(self, node_thresholds: torch.Tensor, edge_thresholds: torch.Tensor):
        # Thresholds are fractions of influence to keep (0.0 to 1.0)
        # Shape: (n_layers,)
        self.node_thresholds = node_thresholds
        self.edge_thresholds = edge_thresholds
        
        # Overrides: force include/exclude specific nodes and edges
        # Sets of node indices (up to 10 nodes)
        self.force_include_nodes: Set[int] = set()
        self.force_exclude_nodes: Set[int] = set()
        
        # Sets of edge tuples (row, col) (up to 100 edges)
        self.force_include_edges: Set[Tuple[int, int]] = set()
        self.force_exclude_edges: Set[Tuple[int, int]] = set()
        
        # Multi-objective attributes for NSGA-II
        # Objective 0: Quality (maximize) = w_completeness * completeness + w_replacement * replacement
        # Objective 1: Complexity (minimize) = w_node * log(nodes) + w_edge * log(edges)
        self.objectives: List[float] = [float('-inf'), float('inf')]  # [quality, complexity]
        
        # NSGA-II ranking attributes
        self.rank: int = -1  # Pareto front rank (0 = best front)
        self.crowding_distance: float = 0.0  # Diversity metric
        
        # Raw metrics (for analysis)
        self.completeness: float = 0.0
        self.replacement: float = 0.0
        self.n_nodes: int = 0
        self.n_edges: int = 0
        
        # Legacy single-objective fitness (for backwards compatibility)
        self.fitness: float = -float('inf')
        
    def clone(self) -> 'Individual':
        """Create a deep copy of this individual."""
        ind = Individual(self.node_thresholds.clone(), self.edge_thresholds.clone())
        ind.force_include_nodes = self.force_include_nodes.copy()
        ind.force_exclude_nodes = self.force_exclude_nodes.copy()
        ind.force_include_edges = self.force_include_edges.copy()
        ind.force_exclude_edges = self.force_exclude_edges.copy()
        ind.objectives = self.objectives.copy()
        ind.rank = self.rank
        ind.crowding_distance = self.crowding_distance
        ind.completeness = self.completeness
        ind.replacement = self.replacement
        ind.n_nodes = self.n_nodes
        ind.n_edges = self.n_edges
        ind.fitness = self.fitness
        return ind
    
    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another.
        
        An individual dominates another if it is:
        - No worse in all objectives
        - Strictly better in at least one objective
        
        Note: Objective 0 (quality) is maximized, Objective 1 (complexity) is minimized.
        """
        dominated = False
        
        # Check objective 0 (quality) - maximize (higher is better)
        if self.objectives[0] < other.objectives[0]:
            return False  # Worse in quality
        if self.objectives[0] > other.objectives[0]:
            dominated = True  # Better in quality
            
        # Check objective 1 (complexity) - minimize (lower is better)
        if self.objectives[1] > other.objectives[1]:
            return False  # Worse in complexity
        if self.objectives[1] < other.objectives[1]:
            dominated = True  # Better in complexity
            
        return dominated
    
    def crowded_comparison(self, other: 'Individual') -> int:
        """
        Compare using NSGA-II crowded comparison operator.
        
        Returns:
            1 if self is better than other
            -1 if other is better than self
            0 if equal
        """
        # Prefer lower rank (better Pareto front)
        if self.rank < other.rank:
            return 1
        if self.rank > other.rank:
            return -1
        
        # Same rank: prefer higher crowding distance (more diverse)
        if self.crowding_distance > other.crowding_distance:
            return 1
        if self.crowding_distance < other.crowding_distance:
            return -1
        
        return 0
    
    def is_evaluated(self) -> bool:
        """Check if objectives have been computed."""
        return self.objectives[0] != float('-inf')
    
    def print_summary(self):
        """Print a summary of this individual's genome and fitness."""
        import math
        mean_node_thresh = self.node_thresholds.mean().item()
        mean_edge_thresh = self.edge_thresholds.mean().item()
        log_nodes = math.log(max(self.n_nodes, 1))
        log_edges = math.log(max(self.n_edges, 1))
        
        print(f"  Rank: {self.rank} | Crowding Distance: {self.crowding_distance:.4f}")
        print(f"  Objectives: Quality={self.objectives[0]:.4f}, Complexity={self.objectives[1]:.4f}")
        print(f"  Completeness: {self.completeness:.4f} | Replacement: {self.replacement:.4f}")
        print(f"  Nodes: {self.n_nodes} (log: {log_nodes:.2f}) | Edges: {self.n_edges} (log: {log_edges:.2f})")
        print(f"  Mean Node Threshold: {mean_node_thresh:.4f} | Mean Edge Threshold: {mean_edge_thresh:.4f}")
        print(f"  Overrides: +{len(self.force_include_nodes)}/-{len(self.force_exclude_nodes)} nodes, "
              f"+{len(self.force_include_edges)}/-{len(self.force_exclude_edges)} edges")
