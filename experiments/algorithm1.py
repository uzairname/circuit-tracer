
import torch
import random
import math
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

from experiments.evaluator import MultiGPUGraphEvaluator
from experiments.individual import Individual
from circuit_tracer.graph import Graph, compute_graph_scores_masked, compute_node_influence, compute_edge_influence, normalize_matrix, compute_influence
from experiments.models import EAHyperparameters


# ==============================================================================
# NSGA-II Core Functions
# ==============================================================================

def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Fast non-dominated sorting algorithm from NSGA-II.
    
    Assigns each individual to a Pareto front (rank).
    Front 0 contains non-dominated solutions (Pareto optimal).
    Front 1 contains solutions dominated only by front 0, etc.
    
    Time complexity: O(M * N^2) where M is number of objectives, N is population size.
    
    Args:
        population: List of evaluated individuals
        
    Returns:
        List of fronts, where each front is a list of individuals
    """
    n = len(population)
    
    if n == 0:
        return []
    
    # S[p] = set of individuals that p dominates
    # n[p] = number of individuals that dominate p
    S = [[] for _ in range(n)]
    domination_count = [0] * n
    
    fronts = [[]]  # fronts[i] = list of individuals in front i
    
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if population[p].dominates(population[q]):
                S[p].append(q)
            elif population[q].dominates(population[p]):
                domination_count[p] += 1
        
        if domination_count[p] == 0:
            population[p].rank = 0
            fronts[0].append(population[p])
  
    # If no non-dominated individuals found (shouldn't happen with valid objectives),
    # put all individuals in front 0 as fallback
    if not fronts[0]:
        for ind in population:
            ind.rank = 0
        fronts[0] = population[:]
        return fronts
    
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p_ind in fronts[i]:
            p = population.index(p_ind)
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    next_front.append(population[q])
        i += 1
        if next_front:
            fronts.append(next_front)
    
    # Remove empty last front if exists
    while fronts and not fronts[-1]:
        fronts.pop()
    
    return fronts


def crowding_distance_assignment(front: List[Individual], n_objectives: int = 2):
    """
    Compute crowding distance for individuals in a front.
    
    Crowding distance measures how close an individual is to its neighbors
    in objective space. Higher distance = more isolated = more valuable for diversity.
    
    Boundary solutions (best/worst in any objective) get infinite distance.
    
    Args:
        front: List of individuals in the same Pareto front
        n_objectives: Number of objectives (default 2)
    """
    n = len(front)
    if n == 0:
        return
    
    # Initialize crowding distance
    for ind in front:
        ind.crowding_distance = 0.0
    
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    for m in range(n_objectives):
        # Sort by objective m
        # Objective 0 (quality): maximize -> sort descending
        # Objective 1 (complexity): minimize -> sort ascending
        if m == 0:
            sorted_front = sorted(front, key=lambda x: x.objectives[m], reverse=True)
        else:
            sorted_front = sorted(front, key=lambda x: x.objectives[m], reverse=False)
        
        # Boundary points get infinite distance
        sorted_front[0].crowding_distance = float('inf')
        sorted_front[-1].crowding_distance = float('inf')
        
        # Compute range for normalization
        f_max = sorted_front[0].objectives[m]
        f_min = sorted_front[-1].objectives[m]
        obj_range = abs(f_max - f_min)
        
        if obj_range < 1e-10:
            continue  # All same value, skip
        
        # Compute crowding distance
        for i in range(1, n - 1):
            if sorted_front[i].crowding_distance != float('inf'):
                distance = abs(sorted_front[i-1].objectives[m] - sorted_front[i+1].objectives[m])
                sorted_front[i].crowding_distance += distance / obj_range


def crowded_comparison_key(ind: Individual) -> Tuple[int, float]:
    """
    Key function for sorting by crowded comparison.
    
    Lower rank is better, higher crowding distance is better.
    Returns tuple for sorting: (rank, -crowding_distance)
    """
    return (ind.rank, -ind.crowding_distance)


def binary_tournament_selection(population: List[Individual]) -> Individual:
    """
    Binary tournament selection using crowded comparison.
    
    Randomly select two individuals and return the better one
    based on Pareto rank and crowding distance.
    """
    a, b = random.sample(population, 2)
    if a.crowded_comparison(b) >= 0:
        return a
    return b



class GraphPrunerEA:
    def __init__(
        self, 
        graph: Graph, 
        hp: EAHyperparameters, 
        gpu_ids: Optional[List[int]] = None,
        max_batch_per_gpu: int = 8,
        verbose: bool = True
    ):
        self.graph = graph
        self.hp = hp
        self.verbose = verbose
        self.device = graph.adjacency_matrix.device
        self.n_layers = graph.cfg.n_layers
        self.gpu_ids = gpu_ids
        self.max_batch_per_gpu = max_batch_per_gpu
        
        # Initialize multi-GPU evaluator for batch operations
        self.multi_gpu_evaluator = MultiGPUGraphEvaluator(
            graph, hp, 
            gpu_ids=gpu_ids, 
            max_batch_per_gpu=max_batch_per_gpu,
            verbose=verbose
        )
        
        # Precompute global influence scores for efficient decoding (single-individual fallback)
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

        # Apply node overrides
        for node_idx in individual.force_include_nodes:
            if 0 <= node_idx < len(node_mask):
                node_mask[node_idx] = True
        for node_idx in individual.force_exclude_nodes:
            if 0 <= node_idx < len(node_mask):
                node_mask[node_idx] = False
        
        # Apply edge overrides
        for edge_idx in individual.force_include_edges:
            row, col = edge_idx
            if 0 <= row < edge_mask.shape[0] and 0 <= col < edge_mask.shape[1]:
                edge_mask[row, col] = True
        for edge_idx in individual.force_exclude_edges:
            row, col = edge_idx
            if 0 <= row < edge_mask.shape[0] and 0 <= col < edge_mask.shape[1]:
                edge_mask[row, col] = False
        
        # Ensure connectivity consistency (optional but good)
        # If a node is pruned, its edges should be pruned.
        # The masked score computation handles this by zeroing out rows/cols of pruned nodes.
        # But explicit masking is cleaner.
        edge_mask[~node_mask, :] = False
        edge_mask[:, ~node_mask] = False
        
        return node_mask, edge_mask

    def evaluate(self, individual: Individual):
        """
        Compute objectives for an individual (NSGA-II multi-objective).
        
        Objectives:
            0: Quality (maximize) = w_completeness * completeness + w_replacement * replacement
            1: Complexity (minimize) = w_node * log(nodes) + w_edge * log(edges)
        """
        node_mask, edge_mask = self.decode(individual)
        
        # Compute graph scores
        replacement, completeness = compute_graph_scores_masked(
            self.graph, node_mask, edge_mask
        )
        
        n_nodes = int(node_mask.sum().item())
        n_edges = int(edge_mask.sum().item())
        
        # Compute objectives
        log_nodes = math.log(max(n_nodes, 1))
        log_edges = math.log(max(n_edges, 1))
        
        # Objective 0: Quality (maximize)
        quality = (
            self.hp.w_completeness * completeness +
            self.hp.w_replacement * replacement
        )
        
        # Objective 1: Complexity (minimize)
        complexity = (
            self.hp.w_complexity_node * log_nodes +
            self.hp.w_complexity_edge * log_edges
        )
        
        individual.objectives = [quality, complexity]
        individual.completeness = completeness
        individual.replacement = replacement
        individual.n_nodes = n_nodes
        individual.n_edges = n_edges
        
        # Legacy fitness (for backwards compatibility)
        individual.fitness = quality - complexity
        
        return individual.objectives

    def decode_batch(self, individuals: List[Individual]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode multiple individuals to binary masks in parallel.
        
        Args:
            individuals: List of individuals to decode
            
        Returns:
            node_masks: (batch_size, N) boolean tensor
            edge_masks: (batch_size, N, N) boolean tensor
        """
        batch_size = len(individuals)
        N = self.graph.adjacency_matrix.shape[0]
        
        # Stack thresholds for vectorized operations
        node_thresholds = torch.stack([ind.node_thresholds for ind in individuals], dim=0)  # (B, n_layers)
        edge_thresholds = torch.stack([ind.edge_thresholds for ind in individuals], dim=0)  # (B, n_layers)
        
        # Initialize masks
        node_masks = torch.zeros(batch_size, N, dtype=torch.bool, device=self.device)
        edge_masks = torch.zeros(batch_size, N, N, dtype=torch.bool, device=self.device)
        
        # Always keep tokens and logits
        n_special = len(self.graph.input_tokens) + len(self.graph.logit_tokens)
        node_masks[:, -n_special:] = True
        
        # Process each layer for nodes
        for layer_idx in range(self.n_layers):
            layer_nodes = (self.node_layers == layer_idx)
            if not layer_nodes.any():
                continue
                
            scores = self.node_influence[layer_nodes]  # (num_layer_nodes,)
            if scores.numel() == 0:
                continue
                
            sorted_scores, _ = torch.sort(scores, descending=True)
            cum_scores = torch.cumsum(sorted_scores, dim=0)
            total_score = cum_scores[-1] + 1e-10
            cum_fractions = cum_scores / total_score
            
            # Get thresholds for this layer across all individuals (B,)
            layer_thresholds = node_thresholds[:, layer_idx]
            
            # Find cutoff indices for each individual
            # searchsorted works on 1D, so we need to handle batch dimension
            cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)  # (B,)
            cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
            
            # Get cutoff scores for each individual
            cutoff_scores = sorted_scores[cutoff_indices]  # (B,)
            
            # Create masks: (B, num_layer_nodes) 
            # scores is broadcast across batch dim, cutoff_scores[:, None] is (B, 1)
            layer_mask = scores.unsqueeze(0) >= cutoff_scores.unsqueeze(1)  # (B, num_layer_nodes)
            
            # Assign to full mask
            node_masks[:, layer_nodes] = layer_mask
        
        # Process each layer for edges
        for layer_idx in range(self.n_layers):
            layer_sources = (self.node_layers == layer_idx)
            if not layer_sources.any():
                continue
                
            current_edge_scores = self.edge_scores[:, layer_sources]  # (N, num_layer_sources)
            if current_edge_scores.numel() == 0:
                continue
                
            flat_scores = current_edge_scores.flatten()
            sorted_scores, _ = torch.sort(flat_scores, descending=True)
            cum_scores = torch.cumsum(sorted_scores, dim=0)
            total_score = cum_scores[-1] + 1e-10
            cum_fractions = cum_scores / total_score
            
            layer_thresholds = edge_thresholds[:, layer_idx]  # (B,)
            cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)
            cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
            cutoff_scores = sorted_scores[cutoff_indices]  # (B,)
            
            # Create masks: (B, N, num_layer_sources)
            layer_mask = current_edge_scores.unsqueeze(0) >= cutoff_scores.view(-1, 1, 1)
            edge_masks[:, :, layer_sources] = layer_mask
        
        # Handle edges from Input Tokens (Layer -1)
        input_sources = (self.node_layers == -1)
        if input_sources.any():
            current_edge_scores = self.edge_scores[:, input_sources]
            if current_edge_scores.numel() > 0:
                flat_scores = current_edge_scores.flatten()
                sorted_scores, _ = torch.sort(flat_scores, descending=True)
                cum_scores = torch.cumsum(sorted_scores, dim=0)
                total_score = cum_scores[-1] + 1e-10
                cum_fractions = cum_scores / total_score
                
                # Use layer 0 threshold for inputs
                layer_thresholds = edge_thresholds[:, 0]
                cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)
                cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
                cutoff_scores = sorted_scores[cutoff_indices]
                
                layer_mask = current_edge_scores.unsqueeze(0) >= cutoff_scores.view(-1, 1, 1)
                edge_masks[:, :, input_sources] = layer_mask
        
        # Apply overrides for each individual
        for batch_idx, individual in enumerate(individuals):
            # Node overrides
            for node_idx in individual.force_include_nodes:
                if 0 <= node_idx < N:
                    node_masks[batch_idx, node_idx] = True
            for node_idx in individual.force_exclude_nodes:
                if 0 <= node_idx < N:
                    node_masks[batch_idx, node_idx] = False
            
            # Edge overrides
            for edge_idx in individual.force_include_edges:
                row, col = edge_idx
                if 0 <= row < N and 0 <= col < N:
                    edge_masks[batch_idx, row, col] = True
            for edge_idx in individual.force_exclude_edges:
                row, col = edge_idx
                if 0 <= row < N and 0 <= col < N:
                    edge_masks[batch_idx, row, col] = False
        
        # Ensure connectivity consistency
        # If a node is pruned, its edges should be pruned
        # Expand node_masks for broadcasting with edge_masks
        node_masks_row = node_masks.unsqueeze(2)  # (B, N, 1)
        node_masks_col = node_masks.unsqueeze(1)  # (B, 1, N)
        edge_masks = edge_masks & node_masks_row & node_masks_col
        
        return node_masks, edge_masks

    def evaluate_batch(self, individuals: List[Individual]) -> List[float]:
        """
        Evaluate fitness for multiple individuals in parallel using multiple GPUs.
        
        Delegates to MultiGPUGraphEvaluator which distributes computation
        across available GPUs.
        
        Args:
            individuals: List of individuals to evaluate
            
        Returns:
            List of fitness values (also updates individual objects in-place)
        """
        return self.multi_gpu_evaluator.evaluate_batch(individuals)

    def _compute_graph_scores_batch(
        self,
        node_masks: torch.Tensor,
        edge_masks: torch.Tensor,
    ) -> Tuple[List[float], List[float], List[int], List[int]]:
        """
        Compute graph scores for a batch of pruned graphs.
        
        Note: This is kept for backwards compatibility but evaluate_batch 
        now uses MultiGPUGraphEvaluator directly.
        
        Args:
            node_masks: (batch_size, N) boolean tensor
            edge_masks: (batch_size, N, N) boolean tensor
            
        Returns:
            replacement_scores: List of replacement scores
            completeness_scores: List of completeness scores  
            n_nodes_list: List of node counts
            n_edges_list: List of edge counts
        """
        batch_size = node_masks.shape[0]
        N = self.graph.adjacency_matrix.shape[0]
        
        n_logits = len(self.graph.logit_tokens)
        n_tokens = len(self.graph.input_tokens)
        n_features = len(self.graph.selected_features)
        error_start = n_features
        error_end = error_start + n_tokens * self.graph.cfg.n_layers
        token_end = error_end + n_tokens
        
        device = self.device
        
        # Create batched masked adjacency matrices
        # (B, N, N)
        adj_matrix = self.graph.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        masked_matrices = adj_matrix * edge_masks.float()
        
        # Zero out rows and columns for pruned nodes
        # node_masks: (B, N) -> need to mask both dims
        node_masks_float = node_masks.float()
        masked_matrices = masked_matrices * node_masks_float.unsqueeze(2)  # mask rows
        masked_matrices = masked_matrices * node_masks_float.unsqueeze(1)  # mask cols
        
        # Logit weights (same for all in batch)
        logit_weights = torch.zeros(N, device=device)
        logit_weights[-n_logits:] = self.graph.logit_probabilities
        logit_weights_batch = logit_weights.unsqueeze(0).expand(batch_size, -1)  # (B, N)
        
        # Normalize matrices: (B, N, N)
        abs_matrices = masked_matrices.abs()
        row_sums = abs_matrices.sum(dim=2, keepdim=True).clamp(min=1e-10)
        normalized_matrices = abs_matrices / row_sums
        
        # Compute influence iteratively for batch
        # current_influence: (B, N)
        # influence = logit_weights @ A + logit_weights @ A^2 + ...
        current_influence = torch.bmm(logit_weights_batch.unsqueeze(1), normalized_matrices).squeeze(1)
        influence = current_influence.clone()
        
        max_iter = 1000
        for iteration in range(max_iter):
            if not current_influence.any():
                break
            current_influence = torch.bmm(current_influence.unsqueeze(1), normalized_matrices).squeeze(1)
            influence = influence + current_influence
        
        # Compute replacement scores
        token_influence = influence[:, error_end:token_end].sum(dim=1)  # (B,)
        error_influence = influence[:, error_start:error_end].sum(dim=1)  # (B,)
        
        total_influence = token_influence + error_influence
        replacement_scores_tensor = torch.where(
            total_influence > 0,
            token_influence / total_influence,
            torch.zeros_like(token_influence)
        )
        
        # Compute completeness scores
        # non_error_fractions: (B, N)
        error_fractions = normalized_matrices[:, :, error_start:error_end].sum(dim=2)
        non_error_fractions = 1 - error_fractions
        
        output_influence = influence + logit_weights_batch  # (B, N)
        output_influence_sum = output_influence.sum(dim=1, keepdim=True).clamp(min=1e-10)
        completeness_scores_tensor = (non_error_fractions * output_influence).sum(dim=1) / output_influence_sum.squeeze(1)
        
        # Count nodes and edges
        n_nodes_tensor = node_masks.sum(dim=1)  # (B,)
        n_edges_tensor = edge_masks.sum(dim=(1, 2))  # (B,)
        
        # Convert to lists
        replacement_scores = replacement_scores_tensor.tolist()
        completeness_scores = completeness_scores_tensor.tolist()
        n_nodes_list = n_nodes_tensor.int().tolist()
        n_edges_list = n_edges_tensor.int().tolist()
        
        return replacement_scores, completeness_scores, n_nodes_list, n_edges_list

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
            
        # Uniform crossover for thresholds
        mask_node = torch.rand(self.n_layers, device=self.device) < 0.5
        mask_edge = torch.rand(self.n_layers, device=self.device) < 0.5
        
        c1_node = torch.where(mask_node, p1.node_thresholds, p2.node_thresholds)
        c2_node = torch.where(mask_node, p2.node_thresholds, p1.node_thresholds)
        
        c1_edge = torch.where(mask_edge, p1.edge_thresholds, p2.edge_thresholds)
        c2_edge = torch.where(mask_edge, p2.edge_thresholds, p1.edge_thresholds)
        
        c1 = Individual(c1_node, c1_edge)
        c2 = Individual(c2_node, c2_edge)
        
        # Crossover overrides: randomly mix from both parents
        # For each override set, take union of both parents and randomly split
        all_force_include_nodes = p1.force_include_nodes | p2.force_include_nodes
        all_force_exclude_nodes = p1.force_exclude_nodes | p2.force_exclude_nodes
        all_force_include_edges = p1.force_include_edges | p2.force_include_edges
        all_force_exclude_edges = p1.force_exclude_edges | p2.force_exclude_edges
        
        # Randomly assign to children
        for node in all_force_include_nodes:
            if random.random() < 0.5:
                c1.force_include_nodes.add(node)
            else:
                c2.force_include_nodes.add(node)
                
        for node in all_force_exclude_nodes:
            if random.random() < 0.5:
                c1.force_exclude_nodes.add(node)
            else:
                c2.force_exclude_nodes.add(node)
                
        for edge in all_force_include_edges:
            if random.random() < 0.5:
                c1.force_include_edges.add(edge)
            else:
                c2.force_include_edges.add(edge)
                
        for edge in all_force_exclude_edges:
            if random.random() < 0.5:
                c1.force_exclude_edges.add(edge)
            else:
                c2.force_exclude_edges.add(edge)
        
        return c1, c2

    def mutate(self, ind: Individual):
        # Mutate thresholds
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
        
        # Mutate overrides
        self._mutate_overrides(ind)
    
    def _mutate_overrides(self, ind: Individual):
        """Mutate override sets by adding, removing, or changing overrides."""
        total_nodes = self.graph.adjacency_matrix.shape[0]
        
        # Mutate node overrides
        if random.random() < self.hp.mutation_rate:
            mutation_type = random.choice(['add_include', 'add_exclude', 'remove_include', 'remove_exclude', 'clear'])
            
            if mutation_type == 'add_include' and len(ind.force_include_nodes) < 10:
                # Add a random node to force include
                # Prefer nodes that are currently excluded or borderline
                candidate = random.randint(0, total_nodes - 1)
                ind.force_include_nodes.add(candidate)
                # Remove from exclude if present
                ind.force_exclude_nodes.discard(candidate)
                
            elif mutation_type == 'add_exclude' and len(ind.force_exclude_nodes) < 10:
                # Add a random node to force exclude
                candidate = random.randint(0, total_nodes - 1)
                ind.force_exclude_nodes.add(candidate)
                # Remove from include if present
                ind.force_include_nodes.discard(candidate)
                
            elif mutation_type == 'remove_include' and ind.force_include_nodes:
                # Remove a random node from force include
                node = random.choice(list(ind.force_include_nodes))
                ind.force_include_nodes.remove(node)
                
            elif mutation_type == 'remove_exclude' and ind.force_exclude_nodes:
                # Remove a random node from force exclude
                node = random.choice(list(ind.force_exclude_nodes))
                ind.force_exclude_nodes.remove(node)
                
            elif mutation_type == 'clear':
                # Clear some overrides
                if random.random() < 0.5:
                    ind.force_include_nodes.clear()
                if random.random() < 0.5:
                    ind.force_exclude_nodes.clear()
        
        # Mutate edge overrides
        if random.random() < self.hp.mutation_rate:
            mutation_type = random.choice(['add_include', 'add_exclude', 'remove_include', 'remove_exclude', 'clear'])
            
            if mutation_type == 'add_include' and len(ind.force_include_edges) < 100:
                # Add a random edge to force include
                row = random.randint(0, total_nodes - 1)
                col = random.randint(0, total_nodes - 1)
                edge = (row, col)
                ind.force_include_edges.add(edge)
                # Remove from exclude if present
                ind.force_exclude_edges.discard(edge)
                
            elif mutation_type == 'add_exclude' and len(ind.force_exclude_edges) < 100:
                # Add a random edge to force exclude
                row = random.randint(0, total_nodes - 1)
                col = random.randint(0, total_nodes - 1)
                edge = (row, col)
                ind.force_exclude_edges.add(edge)
                # Remove from include if present
                ind.force_include_edges.discard(edge)
                
            elif mutation_type == 'remove_include' and ind.force_include_edges:
                # Remove a random edge from force include
                edge = random.choice(list(ind.force_include_edges))
                ind.force_include_edges.remove(edge)
                
            elif mutation_type == 'remove_exclude' and ind.force_exclude_edges:
                # Remove a random edge from force exclude
                edge = random.choice(list(ind.force_exclude_edges))
                ind.force_exclude_edges.remove(edge)
                
            elif mutation_type == 'clear':
                # Clear some overrides
                if random.random() < 0.5:
                    ind.force_include_edges.clear()
                if random.random() < 0.5:
                    ind.force_exclude_edges.clear()

    def run(self, use_batch: bool = True) -> Tuple[List[Individual], Dict[str, List]]:
        """
        Run the NSGA-II evolutionary algorithm.
        
        Args:
            use_batch: If True, use batched parallel evaluation for fitness.
                      If False, evaluate individuals one at a time.
        
        Returns:
            Tuple of (Pareto front individuals, history dict with tracking info)
        """
        if self.verbose:
            print("Starting NSGA-II Multi-Objective Evolutionary Algorithm...")
        population = self.initialize_population()
        
        if self.verbose:
            print(f"Initialized population of size {len(population)}")
        
        # Initialize history tracking
        history = {
            'generation': [],
            'pareto_front_size': [],
            'best_quality': [],         # Best objective 0 (quality)
            'best_complexity': [],      # Best objective 1 (complexity) - lower is better
            'best_n_nodes': [],
            'best_n_edges': [],
            'best_completeness': [],
            'best_replacement': [],
            'hypervolume': [],          # Optional: hypervolume indicator
            'best_fitness': [],         # Best fitness (quality - complexity)
            'median_fitness': [],       # Median fitness across population
            'mean_quality': [],         # Mean quality across population
            'median_quality': [],       # Median quality across population
            'mean_complexity': [],      # Mean complexity across population
            'median_complexity': [],    # Median complexity across population
        }
        
        # Evaluate initial population
        if use_batch:
            self.evaluate_batch(population)
        else:
            for ind in population:
                self.evaluate(ind)
            
        if self.verbose:
            print("Initial evaluation complete.")
            # Debug: print sample objectives
            if population:
                print(f"Sample individual objectives: {population[0].objectives}")
        
        # Initial non-dominated sorting and crowding distance
        fronts = fast_non_dominated_sort(population)
        
        if self.verbose:
            print(f"Number of fronts: {len(fronts)}")
            if fronts:
                print(f"Front 0 size: {len(fronts[0])}")
        
        for front in fronts:
            crowding_distance_assignment(front, n_objectives=self.hp.n_objectives)
        
        # Record initial generation (generation 0)
        pareto_front = fronts[0] if fronts else []
        self._record_history(history, 0, pareto_front, population)
        
        for gen in range(self.hp.n_generations):
            if self.verbose:
                print(f"\n=== Generation {gen+1}/{self.hp.n_generations} ===")
                print(f"Pareto front size: {len(pareto_front)}")
                if pareto_front:
                    best_quality = max(pareto_front, key=lambda x: x.objectives[0])
                    print("Best quality individual:")
                    best_quality.print_summary()
            
            # Create offspring population using binary tournament selection
            offspring = []
            while len(offspring) < self.hp.population_size:
                # Tournament selection using crowded comparison
                p1 = binary_tournament_selection(population)
                p2 = binary_tournament_selection(population)
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                self.mutate(c1)
                self.mutate(c2)
                
                offspring.extend([c1, c2])
            
            # Truncate offspring to population size
            offspring = offspring[:self.hp.population_size]
            
            # Evaluate offspring
            if use_batch:
                to_evaluate = [ind for ind in offspring if not ind.is_evaluated()]
                if to_evaluate:
                    self.evaluate_batch(to_evaluate)
            else:
                for ind in offspring:
                    if not ind.is_evaluated():
                        self.evaluate(ind)
            
            # Combine parent and offspring populations
            combined = population + offspring
            
            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(combined)
            
            # Select next generation using NSGA-II selection
            population = self._nsga2_selection(fronts, self.hp.population_size)
            
            # Update pareto front (front 0 of new population)
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                crowding_distance_assignment(front, n_objectives=self.hp.n_objectives)
            pareto_front = fronts[0] if fronts else []
            
            # Record history
            self._record_history(history, gen + 1, pareto_front, population)
                
        return pareto_front, history
    
    def _nsga2_selection(self, fronts: List[List[Individual]], pop_size: int) -> List[Individual]:
        """
        NSGA-II environmental selection.
        
        Fill the new population by adding complete fronts until 
        the next front would exceed pop_size. Then use crowding 
        distance to select from the last front.
        """
        new_population = []
        
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                # Add entire front
                new_population.extend(front)
            else:
                # Need to select from this front
                remaining = pop_size - len(new_population)
                
                # Compute crowding distance for this front
                crowding_distance_assignment(front, n_objectives=self.hp.n_objectives)
                
                # Sort by crowding distance (descending) and take top remaining
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front[:remaining])
                break
        
        return new_population
    
    def _record_history(self, history: Dict, generation: int, pareto_front: List[Individual], population: List[Individual] = None):
        """Record statistics for a generation."""
        history['generation'].append(generation)
        history['pareto_front_size'].append(len(pareto_front))
        
        if pareto_front:
            best_quality_ind = max(pareto_front, key=lambda x: x.objectives[0])
            best_complexity_ind = min(pareto_front, key=lambda x: x.objectives[1])
            
            history['best_quality'].append(best_quality_ind.objectives[0])
            history['best_complexity'].append(best_complexity_ind.objectives[1])
            history['best_n_nodes'].append(best_complexity_ind.n_nodes)
            history['best_n_edges'].append(best_complexity_ind.n_edges)
            history['best_completeness'].append(best_quality_ind.completeness)
            history['best_replacement'].append(best_quality_ind.replacement)
            
            # Simple hypervolume approximation (area under Pareto front)
            # Reference point: (0, max_complexity * 1.1)
            try:
                hv = self._compute_hypervolume(pareto_front)
                history['hypervolume'].append(hv)
            except:
                history['hypervolume'].append(0.0)
        else:
            # Empty front - shouldn't happen
            history['best_quality'].append(0.0)
            history['best_complexity'].append(float('inf'))
            history['best_n_nodes'].append(0)
            history['best_n_edges'].append(0)
            history['best_completeness'].append(0.0)
            history['best_replacement'].append(0.0)
            history['hypervolume'].append(0.0)
        
        # Compute best and median fitness from the full population
        if population:
            evaluated = [ind for ind in population if ind.is_evaluated()]
            fitness_values = [ind.fitness for ind in evaluated]
            quality_values = [ind.objectives[0] for ind in evaluated]
            complexity_values = [ind.objectives[1] for ind in evaluated]
            
            if fitness_values:
                history['best_fitness'].append(max(fitness_values))
                sorted_fitness = sorted(fitness_values)
                n = len(sorted_fitness)
                if n % 2 == 0:
                    median = (sorted_fitness[n // 2 - 1] + sorted_fitness[n // 2]) / 2
                else:
                    median = sorted_fitness[n // 2]
                history['median_fitness'].append(median)
                
                # Mean and median quality
                history['mean_quality'].append(sum(quality_values) / len(quality_values))
                sorted_quality = sorted(quality_values)
                n = len(sorted_quality)
                if n % 2 == 0:
                    history['median_quality'].append((sorted_quality[n // 2 - 1] + sorted_quality[n // 2]) / 2)
                else:
                    history['median_quality'].append(sorted_quality[n // 2])
                
                # Mean and median complexity
                history['mean_complexity'].append(sum(complexity_values) / len(complexity_values))
                sorted_complexity = sorted(complexity_values)
                n = len(sorted_complexity)
                if n % 2 == 0:
                    history['median_complexity'].append((sorted_complexity[n // 2 - 1] + sorted_complexity[n // 2]) / 2)
                else:
                    history['median_complexity'].append(sorted_complexity[n // 2])
            else:
                history['best_fitness'].append(0.0)
                history['median_fitness'].append(0.0)
                history['mean_quality'].append(0.0)
                history['median_quality'].append(0.0)
                history['mean_complexity'].append(0.0)
                history['median_complexity'].append(0.0)
        else:
            history['best_fitness'].append(0.0)
            history['median_fitness'].append(0.0)
            history['mean_quality'].append(0.0)
            history['median_quality'].append(0.0)
            history['mean_complexity'].append(0.0)
            history['median_complexity'].append(0.0)
    
    def _compute_hypervolume(self, pareto_front: List[Individual]) -> float:
        """
        Compute 2D hypervolume indicator.
        
        Reference point: (0, max_complexity * 1.1)
        Higher hypervolume = better Pareto front.
        """
        if not pareto_front:
            return 0.0
        
        # Sort by quality (objective 0) descending
        sorted_front = sorted(pareto_front, key=lambda x: x.objectives[0], reverse=True)
        
        # Reference point
        ref_quality = 0.0
        ref_complexity = max(ind.objectives[1] for ind in pareto_front) * 1.1 + 1.0
        
        hypervolume = 0.0
        prev_complexity = ref_complexity
        
        for ind in sorted_front:
            quality = ind.objectives[0] - ref_quality
            complexity_diff = prev_complexity - ind.objectives[1]
            
            if quality > 0 and complexity_diff > 0:
                hypervolume += quality * complexity_diff
            
            prev_complexity = ind.objectives[1]
        
        return hypervolume

    def tournament_select(self, population: List[Individual], k: int = 2) -> Individual:
        """
        Binary tournament selection using crowded comparison.
        
        This is kept for backwards compatibility but NSGA-II uses
        binary_tournament_selection function directly.
        """
        return binary_tournament_selection(population)


def create_individual(
    node_thresholds: Union[List[float], torch.Tensor],
    edge_thresholds: Union[List[float], torch.Tensor],
    device: str = "cpu"
) -> Individual:
    """
    Helper function to create an Individual from threshold values.
    
    Args:
        node_thresholds: Per-layer node thresholds (n_layers values between 0-1).
        edge_thresholds: Per-layer edge thresholds (n_layers values between 0-1).
        device: Device to place tensors on.
        
    Returns:
        Individual object.
    """
    if isinstance(node_thresholds, list):
        node_thresholds = torch.tensor(node_thresholds, dtype=torch.float32, device=device)
    if isinstance(edge_thresholds, list):
        edge_thresholds = torch.tensor(edge_thresholds, dtype=torch.float32, device=device)
    
    return Individual(node_thresholds, edge_thresholds)


def run_ea_optimization(
    graph: Graph, 
    verbose: bool = False, 
    use_batch: bool = True, 
    gpu_ids: Optional[List[int]] = None,
    max_batch_per_gpu: int = 8,
    hp: EAHyperparameters = EAHyperparameters(),
) -> Dict:
    """
    Entry point for NSGA-II multi-objective optimization.
    
    Returns a Pareto front of solutions representing different trade-offs
    between quality (completeness + replacement) and complexity (nodes + edges).
    
    Args:
        graph: The computation graph to optimize pruning for.
        verbose: Whether to print progress.
        use_batch: If True, use parallel batch evaluation (faster).
        gpu_ids: List of GPU IDs to use for parallel evaluation. 
                 If None, auto-detects available GPUs.
        max_batch_per_gpu: Maximum individuals to evaluate at once per GPU.
                          Lower values use less memory. Default 8.
        hp: EAHyperparameters configuration.
        
    Returns:
        Dictionary with:
            - pareto_front: List of dicts, each representing a non-dominated solution
            - best_quality: Individual with highest quality score
            - best_complexity: Individual with lowest complexity
            - history: Dict tracking evolution progress
    """
    ea = GraphPrunerEA(
        graph, 
        hp, 
        gpu_ids=gpu_ids, 
        max_batch_per_gpu=max_batch_per_gpu,
        verbose=verbose
    )
    pareto_front, history = ea.run(use_batch=use_batch)
    
    # Handle empty pareto front (shouldn't happen but guard against it)
    if not pareto_front:
        raise ValueError("NSGA-II returned empty Pareto front. Check that graph is valid.")
    
    # Convert Pareto front to serializable format
    pareto_results = []
    for ind in pareto_front:
        pareto_results.append({
            "objectives": ind.objectives.copy(),
            "quality": ind.objectives[0],
            "complexity": ind.objectives[1],
            "completeness": ind.completeness,
            "replacement": ind.replacement,
            "n_nodes": ind.n_nodes,
            "n_edges": ind.n_edges,
            "node_thresholds": ind.node_thresholds.tolist(),
            "edge_thresholds": ind.edge_thresholds.tolist(),
            "rank": ind.rank,
            "crowding_distance": ind.crowding_distance,
        })
    
    # Find extreme solutions
    best_quality_ind = max(pareto_front, key=lambda x: x.objectives[0])
    best_complexity_ind = min(pareto_front, key=lambda x: x.objectives[1])
    
    # Also provide a "balanced" solution (knee point approximation)
    # Normalized objectives to find point closest to ideal
    if len(pareto_front) > 2:
        quality_vals = [ind.objectives[0] for ind in pareto_front]
        complexity_vals = [ind.objectives[1] for ind in pareto_front]
        
        q_min, q_max = min(quality_vals), max(quality_vals)
        c_min, c_max = min(complexity_vals), max(complexity_vals)
        
        q_range = q_max - q_min if q_max > q_min else 1.0
        c_range = c_max - c_min if c_max > c_min else 1.0
        
        # Find solution closest to ideal point (max quality, min complexity)
        best_balanced = min(
            pareto_front,
            key=lambda ind: (
                ((q_max - ind.objectives[0]) / q_range) ** 2 +
                ((ind.objectives[1] - c_min) / c_range) ** 2
            )
        )
    else:
        best_balanced = best_quality_ind
    
    return {
        "pareto_front": pareto_results,
        "pareto_front_size": len(pareto_front),
        
        # Best quality solution (for backwards compatibility)
        "fitness": best_quality_ind.fitness,
        "completeness": best_quality_ind.completeness,
        "replacement": best_quality_ind.replacement,
        "n_nodes": best_quality_ind.n_nodes,
        "n_edges": best_quality_ind.n_edges,
        "node_thresholds": best_quality_ind.node_thresholds.tolist(),
        "edge_thresholds": best_quality_ind.edge_thresholds.tolist(),
        
        # Extreme and balanced solutions
        "best_quality": {
            "quality": best_quality_ind.objectives[0],
            "complexity": best_quality_ind.objectives[1],
            "completeness": best_quality_ind.completeness,
            "replacement": best_quality_ind.replacement,
            "n_nodes": best_quality_ind.n_nodes,
            "n_edges": best_quality_ind.n_edges,
            "node_thresholds": best_quality_ind.node_thresholds.tolist(),
            "edge_thresholds": best_quality_ind.edge_thresholds.tolist(),
        },
        "best_complexity": {
            "quality": best_complexity_ind.objectives[0],
            "complexity": best_complexity_ind.objectives[1],
            "completeness": best_complexity_ind.completeness,
            "replacement": best_complexity_ind.replacement,
            "n_nodes": best_complexity_ind.n_nodes,
            "n_edges": best_complexity_ind.n_edges,
            "node_thresholds": best_complexity_ind.node_thresholds.tolist(),
            "edge_thresholds": best_complexity_ind.edge_thresholds.tolist(),
        },
        "balanced": {
            "quality": best_balanced.objectives[0],
            "complexity": best_balanced.objectives[1],
            "completeness": best_balanced.completeness,
            "replacement": best_balanced.replacement,
            "n_nodes": best_balanced.n_nodes,
            "n_edges": best_balanced.n_edges,
            "node_thresholds": best_balanced.node_thresholds.tolist(),
            "edge_thresholds": best_balanced.edge_thresholds.tolist(),
        },
        
        "history": history
    }
