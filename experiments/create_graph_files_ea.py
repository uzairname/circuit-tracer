from circuit_tracer.graph import Graph
import logging
import os
import time
import torch
from transformers import AutoTokenizer

from circuit_tracer.frontend.graph_models import Metadata, Model, Node, QParams
from circuit_tracer.frontend.utils import add_graph_metadata
from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer.utils.create_graph_files import (
  create_nodes,
  create_used_nodes_and_edges,
  build_model,
  load_graph_data,
)

logger = logging.getLogger(__name__)

def create_graph_files_from_ea_result(
    graph_or_path: Graph | str,
    ea_result: dict,
    slug: str,
    output_path,
    scan=None,
):
    """
    Create graph files from an evolutionary algorithm result.
    
    This function decodes the EA genome (node/edge thresholds and overrides) to create
    a pruned graph, then generates visualization files in the same format as create_graph_files.
    
    Args:
        graph_or_path: Graph object or path to .pt file
        ea_result: Dictionary returned by run_ea_optimization, containing:
            - node_thresholds: Per-layer node threshold values
            - edge_thresholds: Per-layer edge threshold values
            - (optionally) force_include_nodes, force_exclude_nodes, etc.
        slug: Name for the graph files
        output_path: Directory where graph files will be written
        scan: Transcoder scan identifier (if None, uses graph.scan)
    """
    total_start_time = time.time()
    
    # Load graph if needed
    if isinstance(graph_or_path, Graph):
        graph = graph_or_path
    else:
        graph = load_graph_data(graph_or_path)
    
    if os.path.exists(output_path):
        assert os.path.isdir(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)
    
    if scan is None:
        if graph.scan is None:
            raise ValueError(
                "Neither scan nor graph.scan was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        scan = graph.scan
    
    # Decode EA result to get masks
    # Import here to avoid circular dependencies
    import sys
    from pathlib import Path
    
    # Add experiments directory to path if not already there
    repo_root = Path(__file__).parent.parent.parent.parent.parent / "experiments"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from algorithm1 import GraphPrunerEA, create_individual
    from experiments.models import EAHyperparameters
    
    # Create a minimal EA instance just for decoding
    hp = EAHyperparameters()  # Use defaults, we only need decode method
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    
    ea = GraphPrunerEA(graph, hp, verbose=False)
    
    # Create individual from EA result
    best_individual = create_individual(
        node_thresholds=ea_result['node_thresholds'],
        edge_thresholds=ea_result['edge_thresholds'],
        device=device
    )
    
    # Apply overrides if present in result
    if 'force_include_nodes' in ea_result:
        best_individual.force_include_nodes = set(ea_result['force_include_nodes'])
    if 'force_exclude_nodes' in ea_result:
        best_individual.force_exclude_nodes = set(ea_result['force_exclude_nodes'])
    if 'force_include_edges' in ea_result:
        best_individual.force_include_edges = set(tuple(e) for e in ea_result['force_include_edges'])
    if 'force_exclude_edges' in ea_result:
        best_individual.force_exclude_edges = set(tuple(e) for e in ea_result['force_exclude_edges'])
    
    # Decode to get masks
    node_mask, edge_mask = ea.decode(best_individual)
    
    # Compute cumulative scores for nodes (influence)
    cumulative_scores = ea.node_influence
    
    # Move to CPU for file creation
    node_mask = node_mask.cpu()
    edge_mask = edge_mask.cpu()
    cumulative_scores = cumulative_scores.cpu()
    graph.to("cpu")
    
    # Create nodes, edges, and model using existing functions
    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    
    # Calculate effective node/edge threshold from EA result for metadata
    # Use the completeness/replacement scores as proxies
    node_threshold = ea_result.get('completeness', 0.0)
    
    model = build_model(graph, used_nodes, used_edges, slug, scan, node_threshold, tokenizer)
    
    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    add_graph_metadata(model.metadata.model_dump(), output_path)
    logger.info(f"Graph data written to {output_path}")
    
    total_time_ms = (time.time() - total_start_time) * 1000
    logger.info(f"Total execution time: {total_time_ms=:.2f} ms")
    
    return model


