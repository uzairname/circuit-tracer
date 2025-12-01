import torch
from typing import Set, Tuple

class Individual:
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
        
        self.fitness: float = -float('inf')
        self.completeness: float = 0.0
        self.replacement: float = 0.0
        self.n_nodes: int = 0
        self.n_edges: int = 0
        
    def clone(self):
        ind = Individual(self.node_thresholds.clone(), self.edge_thresholds.clone())
        ind.force_include_nodes = self.force_include_nodes.copy()
        ind.force_exclude_nodes = self.force_exclude_nodes.copy()
        ind.force_include_edges = self.force_include_edges.copy()
        ind.force_exclude_edges = self.force_exclude_edges.copy()
        ind.fitness = self.fitness
        ind.completeness = self.completeness
        ind.replacement = self.replacement
        ind.n_nodes = self.n_nodes
        ind.n_edges = self.n_edges
        return ind
    
    def print_summary(self):
        """Print a summary of this individual's genome and fitness."""
        import math
        mean_node_thresh = self.node_thresholds.mean().item()
        mean_edge_thresh = self.edge_thresholds.mean().item()
        log_nodes = math.log(max(self.n_nodes, 1))
        log_edges = math.log(max(self.n_edges, 1))
        print(f"  Fitness: {self.fitness:.4f}")
        print(f"  Completeness: {self.completeness:.4f} | Replacement: {self.replacement:.4f}")
        print(f"  Nodes: {self.n_nodes} (log: {log_nodes:.2f}) | Edges: {self.n_edges} (log: {log_edges:.2f})")
        print(f"  Mean Node Threshold: {mean_node_thresh:.4f} | Mean Edge Threshold: {mean_edge_thresh:.4f}")
        print(f"  Overrides: +{len(self.force_include_nodes)}/-{len(self.force_exclude_nodes)} nodes, "
              f"+{len(self.force_include_edges)}/-{len(self.force_exclude_edges)} edges")
