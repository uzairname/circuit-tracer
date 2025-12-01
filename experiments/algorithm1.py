
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import copy
import time
from tqdm import tqdm

from circuit_tracer.graph import Graph, compute_graph_scores_masked, compute_node_influence, compute_edge_influence, normalize_matrix, compute_influence

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


class MultiGPUGraphEvaluator:
    """
    Evaluator that distributes batch fitness evaluation across multiple GPUs.
    
    Each GPU gets a subset of individuals to evaluate in parallel.
    Memory usage is controlled by processing individuals in sub-batches.
    """
    
    def __init__(
        self, 
        graph: Graph, 
        hp: EAHyperparameters, 
        gpu_ids: Optional[List[int]] = None, 
        max_batch_per_gpu: int = 8,
        verbose: bool = False
    ):
        """
        Initialize multi-GPU evaluator.
        
        Args:
            graph: The computation graph (can be on any device, will be replicated).
            hp: Hyperparameters for fitness calculation.
            gpu_ids: List of GPU device IDs to use. If None, auto-detects available GPUs.
            max_batch_per_gpu: Maximum number of individuals to evaluate at once per GPU.
                              Lower values use less memory but may be slower.
            verbose: Whether to print progress.
        """
        self.hp = hp
        self.verbose = verbose
        self.max_batch_per_gpu = max_batch_per_gpu
        
        # Auto-detect GPUs if not specified
        if gpu_ids is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                gpu_ids = []  # Will fall back to CPU
            else:
                gpu_ids = list(range(n_gpus))
        
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids) if gpu_ids else 1
        
        if self.verbose:
            if self.gpu_ids:
                print(f"MultiGPUGraphEvaluator initialized with {self.n_gpus} GPUs: {self.gpu_ids}")
                print(f"  Max batch per GPU: {self.max_batch_per_gpu}")
            else:
                print("MultiGPUGraphEvaluator: No GPUs available, using CPU")
        
        # Store graph config info (device-agnostic)
        self.n_layers = graph.cfg.n_layers
        self.n_features = len(graph.selected_features)
        self.n_tokens = len(graph.input_tokens)
        self.n_logits = len(graph.logit_tokens)
        self.cfg = graph.cfg
        
        # Replicate graph data to each GPU
        self._replicate_graph_data(graph)
        
        # Precompute layer indices (same across all GPUs, compute once)
        self._precompute_layer_indices(graph)
    
    def _replicate_graph_data(self, graph: Graph):
        """Replicate necessary graph data to each GPU."""
        self.gpu_data = {}
        
        devices = [f"cuda:{gid}" for gid in self.gpu_ids] if self.gpu_ids else ["cpu"]
        
        for device in devices:
            # Move tensors to this device
            adj_matrix = graph.adjacency_matrix.to(device)
            logit_probs = graph.logit_probabilities.to(device)
            
            # Precompute logit weights
            N = adj_matrix.shape[0]
            logit_weights = torch.zeros(N, device=device)
            logit_weights[-self.n_logits:] = logit_probs
            
            # Precompute node influence and edge scores
            node_influence = compute_node_influence(adj_matrix, logit_weights)
            edge_scores = compute_edge_influence(adj_matrix, logit_weights)
            
            self.gpu_data[device] = {
                "adjacency_matrix": adj_matrix,
                "logit_weights": logit_weights,
                "node_influence": node_influence,
                "edge_scores": edge_scores,
            }
    
    def _precompute_layer_indices(self, graph: Graph):
        """Compute layer assignments for nodes (device-agnostic, then replicate)."""
        total_nodes = graph.adjacency_matrix.shape[0]
        
        # Compute on CPU first
        node_layers = torch.full((total_nodes,), -1, dtype=torch.long)
        
        # Features
        if self.n_features > 0:
            feature_indices = graph.selected_features.long().cpu()
            node_layers[:self.n_features] = graph.active_features[feature_indices, 0].long().cpu()
        
        # Error nodes
        error_start = self.n_features
        error_end = error_start + self.n_layers * self.n_tokens
        error_layer_indices = torch.arange(self.n_layers).repeat_interleave(self.n_tokens)
        node_layers[error_start:error_end] = error_layer_indices
        
        # Replicate to each GPU
        devices = [f"cuda:{gid}" for gid in self.gpu_ids] if self.gpu_ids else ["cpu"]
        for device in devices:
            self.gpu_data[device]["node_layers"] = node_layers.to(device)
    
    def _decode_batch_on_device(
        self, 
        node_thresholds: torch.Tensor, 
        edge_thresholds: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode individuals to masks on a specific device.
        
        Args:
            node_thresholds: (batch_size, n_layers) tensor
            edge_thresholds: (batch_size, n_layers) tensor  
            device: Device string (e.g., "cuda:0")
            
        Returns:
            node_masks: (batch_size, N) boolean tensor
            edge_masks: (batch_size, N, N) boolean tensor
        """
        data = self.gpu_data[device]
        node_influence = data["node_influence"]
        edge_scores = data["edge_scores"]
        node_layers = data["node_layers"]
        
        batch_size = node_thresholds.shape[0]
        N = node_influence.shape[0]
        
        # Move thresholds to device
        node_thresholds = node_thresholds.to(device)
        edge_thresholds = edge_thresholds.to(device)
        
        # Initialize masks
        node_masks = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        edge_masks = torch.zeros(batch_size, N, N, dtype=torch.bool, device=device)
        
        # Always keep tokens and logits
        n_special = self.n_tokens + self.n_logits
        node_masks[:, -n_special:] = True
        
        # Process each layer for nodes
        for layer_idx in range(self.n_layers):
            layer_nodes = (node_layers == layer_idx)
            if not layer_nodes.any():
                continue
                
            scores = node_influence[layer_nodes]
            if scores.numel() == 0:
                continue
                
            sorted_scores, _ = torch.sort(scores, descending=True)
            cum_scores = torch.cumsum(sorted_scores, dim=0)
            total_score = cum_scores[-1] + 1e-10
            cum_fractions = cum_scores / total_score
            
            layer_thresholds = node_thresholds[:, layer_idx]
            cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)
            cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
            cutoff_scores = sorted_scores[cutoff_indices]
            
            layer_mask = scores.unsqueeze(0) >= cutoff_scores.unsqueeze(1)
            node_masks[:, layer_nodes] = layer_mask
        
        # Process each layer for edges
        for layer_idx in range(self.n_layers):
            layer_sources = (node_layers == layer_idx)
            if not layer_sources.any():
                continue
                
            current_edge_scores = edge_scores[:, layer_sources]
            if current_edge_scores.numel() == 0:
                continue
                
            flat_scores = current_edge_scores.flatten()
            sorted_scores, _ = torch.sort(flat_scores, descending=True)
            cum_scores = torch.cumsum(sorted_scores, dim=0)
            total_score = cum_scores[-1] + 1e-10
            cum_fractions = cum_scores / total_score
            
            layer_thresholds = edge_thresholds[:, layer_idx]
            cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)
            cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
            cutoff_scores = sorted_scores[cutoff_indices]
            
            layer_mask = current_edge_scores.unsqueeze(0) >= cutoff_scores.view(-1, 1, 1)
            edge_masks[:, :, layer_sources] = layer_mask
        
        # Handle edges from Input Tokens (Layer -1)
        input_sources = (node_layers == -1)
        if input_sources.any():
            current_edge_scores = edge_scores[:, input_sources]
            if current_edge_scores.numel() > 0:
                flat_scores = current_edge_scores.flatten()
                sorted_scores, _ = torch.sort(flat_scores, descending=True)
                cum_scores = torch.cumsum(sorted_scores, dim=0)
                total_score = cum_scores[-1] + 1e-10
                cum_fractions = cum_scores / total_score
                
                layer_thresholds = edge_thresholds[:, 0]
                cutoff_indices = torch.searchsorted(cum_fractions, layer_thresholds)
                cutoff_indices = torch.clamp(cutoff_indices, max=len(sorted_scores) - 1)
                cutoff_scores = sorted_scores[cutoff_indices]
                
                layer_mask = current_edge_scores.unsqueeze(0) >= cutoff_scores.view(-1, 1, 1)
                edge_masks[:, :, input_sources] = layer_mask
        
        # Ensure connectivity consistency
        node_masks_row = node_masks.unsqueeze(2)
        node_masks_col = node_masks.unsqueeze(1)
        edge_masks = edge_masks & node_masks_row & node_masks_col
        
        return node_masks, edge_masks
    
    def _compute_scores_on_device(
        self,
        node_masks: torch.Tensor,
        edge_masks: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute graph scores for a batch on a specific device.
        
        Returns tensors (not lists) for efficiency.
        """
        data = self.gpu_data[device]
        adj_matrix = data["adjacency_matrix"]
        logit_weights = data["logit_weights"]
        
        batch_size = node_masks.shape[0]
        N = adj_matrix.shape[0]
        
        error_start = self.n_features
        error_end = error_start + self.n_tokens * self.n_layers
        token_end = error_end + self.n_tokens
        
        # Create batched masked adjacency matrices
        adj_expanded = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        masked_matrices = adj_expanded * edge_masks.float()
        
        # Zero out rows and columns for pruned nodes
        node_masks_float = node_masks.float()
        masked_matrices = masked_matrices * node_masks_float.unsqueeze(2)
        masked_matrices = masked_matrices * node_masks_float.unsqueeze(1)
        
        # Logit weights batch
        logit_weights_batch = logit_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Normalize matrices
        abs_matrices = masked_matrices.abs()
        row_sums = abs_matrices.sum(dim=2, keepdim=True).clamp(min=1e-10)
        normalized_matrices = abs_matrices / row_sums
        
        # Compute influence iteratively
        current_influence = torch.bmm(logit_weights_batch.unsqueeze(1), normalized_matrices).squeeze(1)
        influence = current_influence.clone()
        
        max_iter = 1000
        for _ in range(max_iter):
            if not current_influence.any():
                break
            current_influence = torch.bmm(current_influence.unsqueeze(1), normalized_matrices).squeeze(1)
            influence = influence + current_influence
        
        # Compute replacement scores
        token_influence = influence[:, error_end:token_end].sum(dim=1)
        error_influence = influence[:, error_start:error_end].sum(dim=1)
        
        total_influence = token_influence + error_influence
        replacement_scores = torch.where(
            total_influence > 0,
            token_influence / total_influence,
            torch.zeros_like(token_influence)
        )
        
        # Compute completeness scores
        error_fractions = normalized_matrices[:, :, error_start:error_end].sum(dim=2)
        non_error_fractions = 1 - error_fractions
        
        output_influence = influence + logit_weights_batch
        output_influence_sum = output_influence.sum(dim=1, keepdim=True).clamp(min=1e-10)
        completeness_scores = (non_error_fractions * output_influence).sum(dim=1) / output_influence_sum.squeeze(1)
        
        # Count nodes and edges
        n_nodes = node_masks.sum(dim=1)
        n_edges = edge_masks.sum(dim=(1, 2))
        
        return replacement_scores, completeness_scores, n_nodes, n_edges
    
    def evaluate_batch(self, individuals: List[Individual]) -> List[float]:
        """
        Evaluate fitness for multiple individuals using multiple GPUs.
        
        Splits the population across GPUs, evaluates in parallel, and 
        gathers results.
        
        Args:
            individuals: List of individuals to evaluate.
            
        Returns:
            List of fitness values (also updates individual objects in-place).
        """
        if len(individuals) == 0:
            return []
        
        # Stack all thresholds
        node_thresholds = torch.stack([ind.node_thresholds for ind in individuals], dim=0)
        edge_thresholds = torch.stack([ind.edge_thresholds for ind in individuals], dim=0)
        
        batch_size = len(individuals)
        devices = [f"cuda:{gid}" for gid in self.gpu_ids] if self.gpu_ids else ["cpu"]
        
        # Split indices across GPUs
        chunk_size = (batch_size + self.n_gpus - 1) // self.n_gpus
        
        # Results storage
        all_replacement = []
        all_completeness = []
        all_n_nodes = []
        all_n_edges = []
        
        for gpu_idx, device in enumerate(devices):
            start_idx = gpu_idx * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)
            
            if start_idx >= batch_size:
                break
            
            # Get chunk for this GPU
            chunk_node_thresh = node_thresholds[start_idx:end_idx]
            chunk_edge_thresh = edge_thresholds[start_idx:end_idx]
            
            # Process in sub-batches to limit memory usage
            gpu_replacement, gpu_completeness, gpu_n_nodes, gpu_n_edges = \
                self._evaluate_chunk_on_device(chunk_node_thresh, chunk_edge_thresh, device)
            
            # Collect results
            all_replacement.append(gpu_replacement)
            all_completeness.append(gpu_completeness)
            all_n_nodes.append(gpu_n_nodes)
            all_n_edges.append(gpu_n_edges)
        
        # Concatenate results
        replacement_scores = torch.cat(all_replacement, dim=0).tolist()
        completeness_scores = torch.cat(all_completeness, dim=0).tolist()
        n_nodes_list = torch.cat(all_n_nodes, dim=0).int().tolist()
        n_edges_list = torch.cat(all_n_edges, dim=0).int().tolist()
        
        # Calculate fitness and update individuals
        fitness_values = []
        for i, ind in enumerate(individuals):
            replacement = replacement_scores[i]
            completeness = completeness_scores[i]
            n_nodes = n_nodes_list[i]
            n_edges = n_edges_list[i]
            
            fitness = (
                self.hp.w_completeness * completeness +
                self.hp.w_replacement * replacement -
                self.hp.w_complexity_node * n_nodes -
                self.hp.w_complexity_edge * n_edges
            )
            
            ind.fitness = fitness
            ind.completeness = completeness
            ind.replacement = replacement
            ind.n_nodes = n_nodes
            ind.n_edges = n_edges
            
            fitness_values.append(fitness)
        
        return fitness_values
    
    def _evaluate_chunk_on_device(
        self,
        node_thresholds: torch.Tensor,
        edge_thresholds: torch.Tensor,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a chunk of individuals on a single device, processing in 
        sub-batches to limit memory usage.
        
        Args:
            node_thresholds: (chunk_size, n_layers) tensor
            edge_thresholds: (chunk_size, n_layers) tensor
            device: Device string
            
        Returns:
            Tuple of CPU tensors: (replacement, completeness, n_nodes, n_edges)
        """
        chunk_size = node_thresholds.shape[0]
        
        # Process in sub-batches
        sub_replacement = []
        sub_completeness = []
        sub_n_nodes = []
        sub_n_edges = []
        
        for sub_start in range(0, chunk_size, self.max_batch_per_gpu):
            sub_end = min(sub_start + self.max_batch_per_gpu, chunk_size)
            
            sub_node_thresh = node_thresholds[sub_start:sub_end]
            sub_edge_thresh = edge_thresholds[sub_start:sub_end]
            
            # Decode on this GPU
            node_masks, edge_masks = self._decode_batch_on_device(
                sub_node_thresh, sub_edge_thresh, device
            )
            
            # Compute scores on this GPU
            replacement, completeness, n_nodes, n_edges = self._compute_scores_on_device(
                node_masks, edge_masks, device
            )
            
            # Move to CPU immediately to free GPU memory
            sub_replacement.append(replacement.cpu())
            sub_completeness.append(completeness.cpu())
            sub_n_nodes.append(n_nodes.cpu())
            sub_n_edges.append(n_edges.cpu())
            
            # Free GPU memory
            del node_masks, edge_masks, replacement, completeness, n_nodes, n_edges
            torch.cuda.empty_cache()
        
        # Concatenate sub-batch results
        return (
            torch.cat(sub_replacement, dim=0),
            torch.cat(sub_completeness, dim=0),
            torch.cat(sub_n_nodes, dim=0),
            torch.cat(sub_n_edges, dim=0)
        )


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

    def run(self, use_batch: bool = True) -> Individual:
        """
        Run the evolutionary algorithm.
        
        Args:
            use_batch: If True, use batched parallel evaluation for fitness.
                      If False, evaluate individuals one at a time.
        
        Returns:
            Best individual found.
        """
        if self.verbose:
            print("Starting Evolutionary Algorithm...")
        population = self.initialize_population()
        
        if self.verbose:
            print(f"Initialized population of size {len(population)}")
        
        # Evaluate initial population
        if use_batch:
            self.evaluate_batch(population)
        else:
            for ind in population:
                if self.verbose:
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
            
            # Evaluate new individuals
            if use_batch:
                # Collect individuals that need evaluation
                to_evaluate = [ind for ind in population if ind.fitness == -float('inf')]
                if to_evaluate:
                    self.evaluate_batch(to_evaluate)
            else:
                for ind in population:
                    if ind.fitness == -float('inf'):  # Only evaluate new
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


def evaluate_individuals_parallel(
    graph: Graph,
    individuals: List[Individual],
    hp: Optional[EAHyperparameters] = None,
    gpu_ids: Optional[List[int]] = None,
    max_batch_per_gpu: int = 8,
    verbose: bool = False
) -> List[Dict]:
    """
    Evaluate fitness for multiple individuals in parallel using multiple GPUs.
    
    This is a standalone function that can be used to evaluate a batch of
    individuals without running the full EA. Useful for:
    - Custom optimization loops
    - Hyperparameter tuning
    - Analyzing specific threshold configurations
    
    Args:
        graph: The computation graph to prune.
        individuals: List of Individual objects to evaluate.
        hp: Hyperparameters (optional, uses defaults if not provided).
        gpu_ids: List of GPU IDs to use. If None, auto-detects available GPUs.
        max_batch_per_gpu: Maximum individuals per GPU sub-batch.
        verbose: Whether to print progress.
        
    Returns:
        List of dictionaries with fitness metrics for each individual.
    """
    if hp is None:
        hp = EAHyperparameters()
    
    evaluator = MultiGPUGraphEvaluator(
        graph, hp, 
        gpu_ids=gpu_ids, 
        max_batch_per_gpu=max_batch_per_gpu,
        verbose=verbose
    )
    evaluator.evaluate_batch(individuals)
    
    results = []
    for ind in individuals:
        results.append({
            "fitness": ind.fitness,
            "completeness": ind.completeness,
            "replacement": ind.replacement,
            "n_nodes": ind.n_nodes,
            "n_edges": ind.n_edges,
            "node_thresholds": ind.node_thresholds.tolist(),
            "edge_thresholds": ind.edge_thresholds.tolist()
        })
    
    return results


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
    **kwargs
):
    """
    Entry point for hyperparameter optimization.
    
    Args:
        graph: The computation graph to optimize pruning for.
        verbose: Whether to print progress.
        use_batch: If True, use parallel batch evaluation (faster).
        gpu_ids: List of GPU IDs to use for parallel evaluation. 
                 If None, auto-detects available GPUs.
        max_batch_per_gpu: Maximum individuals to evaluate at once per GPU.
                          Lower values use less memory. Default 8.
        **kwargs: Override default EAHyperparameters.
        
    Returns:
        Dictionary with best individual's results.
    """
    hp = EAHyperparameters(**kwargs)
    ea = GraphPrunerEA(
        graph, hp, 
        gpu_ids=gpu_ids, 
        max_batch_per_gpu=max_batch_per_gpu,
        verbose=verbose
    )
    best_ind = ea.run(use_batch=use_batch)
    
    return {
        "fitness": best_ind.fitness,
        "completeness": best_ind.completeness,
        "replacement": best_ind.replacement,
        "n_nodes": best_ind.n_nodes,
        "n_edges": best_ind.n_edges,
        "node_thresholds": best_ind.node_thresholds.tolist(),
        "edge_thresholds": best_ind.edge_thresholds.tolist()
    }
