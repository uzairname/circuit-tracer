
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import copy
import time
from tqdm import tqdm

from circuit_tracer.graph import Graph, compute_graph_scores_masked, compute_node_influence, compute_edge_influence, normalize_matrix

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
    w_complexity_node: float = 0.0001 # Penalty per node
    w_complexity_edge: float = 0.00001 # Penalty per edge
    
    # Constraints / Bounds
    min_threshold: float = 0.0
    max_threshold: float = 1.0

class Individual:
    def __init__(self, node_thresholds: torch.Tensor, edge_thresholds: torch.Tensor):
        # Thresholds are fractions of influence to keep (0.0 to 1.0)
        # Shape: (n_layers,)
        self.node_thresholds = node_thresholds
        self.edge_thresholds = edge_thresholds
        
        self.fitness: float = -float('inf')
        self.completeness: float = 0.0
        self.replacement: float = 0.0
        self.n_nodes: int = 0
        self.n_edges: int = 0
        
    def clone(self):
        ind = Individual(self.node_thresholds.clone(), self.edge_thresholds.clone())
        ind.fitness = self.fitness
        ind.completeness = self.completeness
        ind.replacement = self.replacement
        ind.n_nodes = self.n_nodes
        ind.n_edges = self.n_edges
        return ind

class GraphPrunerEA:
    def __init__(self, graph: Graph, hp: EAHyperparameters, verbose: bool = True):
        self.graph = graph
        self.hp = hp
        self.verbose = verbose
        self.device = graph.adjacency_matrix.device
        self.n_layers = graph.cfg.n_layers
        
        # Precompute global influence scores for efficient decoding
        self._precompute_scores()
        
        # Identify layers for all nodes and edges
        self._identify_layers()
        
    def _precompute_scores(self):
        """Precompute node and edge influence scores on the full graph."""
        if self.verbose:
            print("Precomputing graph influence scores...")
            
        n_logits = len(self.graph.logit_tokens)
        logit_weights = torch.zeros(
            self.graph.adjacency_matrix.shape[0], device=self.device
        )
        logit_weights[-n_logits:] = self.graph.logit_probabilities
        
        # Node influence
        self.node_influence = compute_node_influence(self.graph.adjacency_matrix, logit_weights)
        
        # Edge influence (using full graph approximation for selection)
        # We use the edge scores from the full graph to decide which edges to keep.
        # This is an approximation because strictly edge influence depends on pruned nodes,
        # but it's necessary for efficient encoding/decoding.
        # We use the same logic as prune_graph but without the loop.
        self.edge_scores = compute_edge_influence(self.graph.adjacency_matrix, logit_weights)
        
    def _identify_layers(self):
        """Map every node and edge to a layer index."""
        # selected_features contains indices into active_features for features in the adjacency matrix
        # The adjacency matrix is sized for selected_features, not active_features
        n_features = len(self.graph.selected_features)
        n_tokens = len(self.graph.input_tokens)
        n_logits = len(self.graph.logit_tokens)
        total_nodes = self.graph.adjacency_matrix.shape[0]
        
        # Store dimensions for later use
        self.n_features = n_features
        self.n_tokens = n_tokens
        self.n_logits = n_logits
        
        # Node Layers
        # Initialize with -1 (for tokens/logits which we might treat separately or as specific layers)
        self.node_layers = torch.full((total_nodes,), -1, dtype=torch.long, device=self.device)
        
        # 1. Features
        # Get layer info from active_features using selected_features as indices
        if n_features > 0:
            # selected_features[i] is the index into active_features for the i-th feature node
            feature_indices = self.graph.selected_features.long()
            self.node_layers[:n_features] = self.graph.active_features[feature_indices, 0].long()
            
        # 2. Error nodes
        # Ordered by layer (outer) then position (inner)
        # n_layers * n_tokens nodes
        error_start = n_features
        error_end = error_start + self.n_layers * n_tokens
        
        error_layer_indices = torch.arange(self.n_layers, device=self.device).repeat_interleave(n_tokens)
        self.node_layers[error_start:error_end] = error_layer_indices
        
        # 3. Input Tokens (Embeddings)
        # Usually treated as Layer -1 or Input. Let's map them to Layer 0 for thresholding purposes 
        # or handle them separately. The paper says they are not pruned.
        # We will mark them as -1 and ensure they are always kept.
        
        # 4. Logits
        # Always kept.
        
        # Edge Layers
        # We assign edges to the layer of their SOURCE node.
        # This allows `edge_thresholds[l]` to control edges originating from layer `l`.
        # We need to handle edges from inputs (layer -1).
        
        # Create a dense tensor of edge layers matching adjacency matrix shape? No, too big.
        # We only care about existing edges.
        # But `edge_scores` is dense (or same shape as adj).
        # We can use broadcasting.
        # `self.node_layers` shape is (N,).
        # Edge (i, j) source is j. So layer is `node_layers[j]`.
        # We can use `self.node_layers` directly during decoding.
        
    def decode(self, individual: Individual) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert individual (thresholds) to binary masks.
        
        Returns:
            node_mask: (N,) boolean tensor
            edge_mask: (N, N) boolean tensor
        """
        # 1. Node Mask
        # For each layer, find the score cutoff corresponding to the threshold fraction
        # We can do this in a vectorized way or loop over layers. Loop is fine for small n_layers (12-18).
        
        node_mask = torch.zeros_like(self.node_influence, dtype=torch.bool)
        
        # Always keep tokens and logits
        n_special = len(self.graph.input_tokens) + len(self.graph.logit_tokens)
        node_mask[-n_special:] = True
        
        # For each layer, apply threshold
        for layer_idx in range(self.n_layers):
            # Get nodes in this layer
            layer_nodes = (self.node_layers == layer_idx)
            if not layer_nodes.any():
                continue
                
            scores = self.node_influence[layer_nodes]
            threshold_val = individual.node_thresholds[layer_idx]
            
            # Calculate cutoff score
            # We want to keep nodes that contribute to top X% of influence in this layer
            if scores.numel() > 0:
                sorted_scores, _ = torch.sort(scores, descending=True)
                cum_scores = torch.cumsum(sorted_scores, dim=0)
                total_score = cum_scores[-1] + 1e-10
                cum_fractions = cum_scores / total_score
                
                # Find index where cumulative fraction >= threshold
                # If threshold is 0.8, we keep nodes up to 0.8
                idx = torch.searchsorted(cum_fractions, threshold_val)
                if idx < len(scores):
                    cutoff = sorted_scores[idx]
                    # Keep nodes >= cutoff
                    node_mask[layer_nodes] = (self.node_influence[layer_nodes] >= cutoff)
                else:
                    # Keep all if threshold is 1.0 (or close)
                    node_mask[layer_nodes] = True
                    
        # 2. Edge Mask
        # Similar logic, but per source layer
        # edge_scores is (N, N)
        # We can iterate over source layers (columns)
        
        edge_mask = torch.zeros_like(self.edge_scores, dtype=torch.bool)
        
        for layer_idx in range(self.n_layers):
            # Identify columns (sources) belonging to this layer
            layer_sources = (self.node_layers == layer_idx)
            if not layer_sources.any():
                continue
                
            # Extract scores for edges originating from this layer
            # Shape: (N, num_layer_sources)
            # We flatten to find the threshold for this layer's outgoing edges
            current_edge_scores = self.edge_scores[:, layer_sources]
            
            threshold_val = individual.edge_thresholds[layer_idx]
            
            if current_edge_scores.numel() > 0:
                flat_scores = current_edge_scores.flatten()
                sorted_scores, _ = torch.sort(flat_scores, descending=True)
                cum_scores = torch.cumsum(sorted_scores, dim=0)
                total_score = cum_scores[-1] + 1e-10
                cum_fractions = cum_scores / total_score
                
                idx = torch.searchsorted(cum_fractions, threshold_val)
                if idx < len(flat_scores):
                    cutoff = sorted_scores[idx]
                    # Create mask for this layer's columns
                    layer_mask = (current_edge_scores >= cutoff)
                    # Assign back to full mask
                    edge_mask[:, layer_sources] = layer_mask
                else:
                    edge_mask[:, layer_sources] = True
                    
        # Handle edges from Input Tokens (Layer -1)
        # We can use a default threshold or keep them all. 
        # Usually we want to prune them too. Let's use the first layer's threshold or a separate one?
        # For simplicity, let's keep all edges from input tokens for now, or use Layer 0 threshold.
        # Let's use Layer 0 threshold for inputs.
        input_sources = (self.node_layers == -1)
        if input_sources.any():
             # Use Layer 0 threshold
            threshold_val = individual.edge_thresholds[0]
            current_edge_scores = self.edge_scores[:, input_sources]
            flat_scores = current_edge_scores.flatten()
            sorted_scores, _ = torch.sort(flat_scores, descending=True)
            cum_scores = torch.cumsum(sorted_scores, dim=0)
            total_score = cum_scores[-1] + 1e-10
            cum_fractions = cum_scores / total_score
            idx = torch.searchsorted(cum_fractions, threshold_val)
            if idx < len(flat_scores):
                cutoff = sorted_scores[idx]
                edge_mask[:, input_sources] = (current_edge_scores >= cutoff)
            else:
                edge_mask[:, input_sources] = True

        # Ensure connectivity consistency (optional but good)
        # If a node is pruned, its edges should be pruned.
        # The masked score computation handles this by zeroing out rows/cols of pruned nodes.
        # But explicit masking is cleaner.
        edge_mask[~node_mask, :] = False
        edge_mask[:, ~node_mask] = False
        
        return node_mask, edge_mask

    def evaluate(self, individual: Individual):
        """Compute fitness for an individual."""
        node_mask, edge_mask = self.decode(individual)
        
        # Compute graph scores
        # This is the expensive part
        replacement, completeness = compute_graph_scores_masked(
            self.graph, node_mask, edge_mask
        )
        
        n_nodes = int(node_mask.sum().item())
        n_edges = int(edge_mask.sum().item())
        
        # Calculate fitness
        # Maximize replacement & completeness, Minimize nodes & edges
        fitness = (
            self.hp.w_completeness * completeness +
            self.hp.w_replacement * replacement -
            self.hp.w_complexity_node * n_nodes -
            self.hp.w_complexity_edge * n_edges
        )
        
        individual.fitness = fitness
        individual.completeness = completeness
        individual.replacement = replacement
        individual.n_nodes = n_nodes
        individual.n_edges = n_edges
        
        return fitness

    def initialize_population(self) -> List[Individual]:
        pop = []
        for _ in range(self.hp.population_size):
            # Random initialization
            # Bias towards higher thresholds (0.5 - 1.0) as we want to keep some structure
            node_thresh = torch.rand(self.n_layers, device=self.device) * 0.5 + 0.5
            edge_thresh = torch.rand(self.n_layers, device=self.device) * 0.5 + 0.5
            pop.append(Individual(node_thresh, edge_thresh))
        return pop

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.hp.crossover_rate:
            return p1.clone(), p2.clone()
            
        # Uniform crossover
        mask_node = torch.rand(self.n_layers, device=self.device) < 0.5
        mask_edge = torch.rand(self.n_layers, device=self.device) < 0.5
        
        c1_node = torch.where(mask_node, p1.node_thresholds, p2.node_thresholds)
        c2_node = torch.where(mask_node, p2.node_thresholds, p1.node_thresholds)
        
        c1_edge = torch.where(mask_edge, p1.edge_thresholds, p2.edge_thresholds)
        c2_edge = torch.where(mask_edge, p2.edge_thresholds, p1.edge_thresholds)
        
        return Individual(c1_node, c1_edge), Individual(c2_node, c2_edge)

    def mutate(self, ind: Individual):
        if random.random() < self.hp.mutation_rate:
            # Gaussian mutation
            noise_node = torch.randn_like(ind.node_thresholds) * self.hp.mutation_sigma
            noise_edge = torch.randn_like(ind.edge_thresholds) * self.hp.mutation_sigma
            
            ind.node_thresholds = torch.clamp(
                ind.node_thresholds + noise_node, 
                self.hp.min_threshold, 
                self.hp.max_threshold
            )
            ind.edge_thresholds = torch.clamp(
                ind.edge_thresholds + noise_edge, 
                self.hp.min_threshold, 
                self.hp.max_threshold
            )

    def run(self) -> Individual:
        if (self.verbose):
            print("Starting Evolutionary Algorithm...")
        population = self.initialize_population()
        
        if self.verbose:
            print(f"Initialized population of size {len(population)}")
        
        # Evaluate initial population
        for ind in population:
            if (self.verbose):
                print(f"evaluating individual {ind}")
            self.evaluate(ind)
            
        if self.verbose:
            print("Initial evaluation complete.")
            
        best_ind = max(population, key=lambda x: x.fitness)
        
        for gen in range(self.hp.n_generations):
            if self.verbose:
                print(f"Generation {gen+1}/{self.hp.n_generations} | Best Fitness: {best_ind.fitness:.4f} | Nodes: {best_ind.n_nodes}")
            
            new_pop = []
            
            # Elitism: keep best
            new_pop.append(best_ind.clone())
            
            while len(new_pop) < self.hp.population_size:
                # Tournament selection
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                self.mutate(c1)
                self.mutate(c2)
                
                new_pop.extend([c1, c2])
                
            # Truncate to pop size
            population = new_pop[:self.hp.population_size]
            
            # Evaluate
            for ind in population:
                if ind.fitness == -float('inf'): # Only evaluate new
                    self.evaluate(ind)
            
            # Update best
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_ind.fitness:
                best_ind = current_best.clone()
                
        return best_ind

    def tournament_select(self, population, k=3):
        candidates = random.sample(population, k)
        return max(candidates, key=lambda x: x.fitness)

import random

def run_ea_optimization(graph: Graph, verbose=False, **kwargs):
    """
    Entry point for hyperparameter optimization.
    kwargs can override default EAHyperparameters.
    """
    hp = EAHyperparameters(**kwargs)
    ea = GraphPrunerEA(graph, hp, verbose=verbose)
    best_ind = ea.run()
    
    return {
        "fitness": best_ind.fitness,
        "completeness": best_ind.completeness,
        "replacement": best_ind.replacement,
        "n_nodes": best_ind.n_nodes,
        "n_edges": best_ind.n_edges,
        "node_thresholds": best_ind.node_thresholds.tolist(),
        "edge_thresholds": best_ind.edge_thresholds.tolist()
    }
