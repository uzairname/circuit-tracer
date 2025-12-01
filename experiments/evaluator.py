import torch
from typing import List, Optional, Tuple, Dict
from circuit_tracer.graph import Graph, compute_edge_influence, compute_node_influence
from experiments.models import EAHyperparameters
from experiments.individual import Individual


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
        
        # Calculate objectives (NSGA-II multi-objective) and update individuals
        import math
        fitness_values = []
        for i, ind in enumerate(individuals):
            replacement = replacement_scores[i]
            completeness = completeness_scores[i]
            n_nodes = n_nodes_list[i]
            n_edges = n_edges_list[i]
            
            # Use log to handle vastly different magnitudes (edges ~ nodes^2)
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
            
            ind.objectives = [quality, complexity]
            ind.completeness = completeness
            ind.replacement = replacement
            ind.n_nodes = n_nodes
            ind.n_edges = n_edges
            
            # Legacy single-objective fitness (for backwards compatibility)
            ind.fitness = quality - complexity
            
            fitness_values.append(ind.fitness)
        
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