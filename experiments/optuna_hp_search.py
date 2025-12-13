"""
Hyperparameter optimization for the evolutionary algorithm using Optuna.

This script runs Optuna trials to find optimal hyperparameters for the NSGA-II
evolutionary algorithm that prunes computation graphs.

Usage:
    python optuna_hp_search.py --graph-path graphs/war.pt --n-trials 50
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import json

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from circuit_tracer.utils.create_graph_files import load_graph_data
from algorithm1 import run_ea_optimization
from models import EAHyperparameters


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, graph, base_hp: Dict) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        graph: The computation graph to optimize
        base_hp: Base hyperparameters (fixed values)
        
    Returns:
        Metric to optimize (we'll use balanced solution quality)
    """
    # Sample hyperparameters
    hp = EAHyperparameters(
        population_size=trial.suggest_int("population_size", 20, 100, step=10),
        n_generations=base_hp.get("n_generations", 50),  # Fixed to save time
        mutation_rate=trial.suggest_float("mutation_rate", 0.05, 0.5),
        crossover_rate=trial.suggest_float("crossover_rate", 0.5, 0.95),
        mutation_sigma=trial.suggest_float("mutation_sigma", 0.05, 0.3),
        
        # Objective weights - these are critical for the quality/complexity tradeoff
        w_completeness=trial.suggest_float("w_completeness", 0.1, 2.0),
        w_replacement=trial.suggest_float("w_replacement", 0.1, 2.0),
        w_complexity_node=trial.suggest_float("w_complexity_node", 1.0, 15.0),
        w_complexity_edge=trial.suggest_float("w_complexity_edge", 0.5, 10.0),
    )
    
    # Run EA optimization
    try:
        result = run_ea_optimization(
            graph=graph,
            verbose=False,
            use_batch=True,
            max_batch_per_gpu=base_hp.get("max_batch_per_gpu", 8),
            hp=hp,
        )
        
        # Use balanced solution as the primary metric
        balanced = result['balanced']
        
        # Compute a composite score that balances quality and simplicity
        # Higher quality is better, lower complexity is better
        quality = balanced['quality']
        complexity = balanced['complexity']
        
        # Normalize and combine (you can adjust this formula based on your preferences)
        # We want high quality and low complexity
        score = quality - 0.1 * complexity
        
        # Log intermediate results
        trial.set_user_attr("quality", quality)
        trial.set_user_attr("complexity", complexity)
        trial.set_user_attr("completeness", balanced['completeness'])
        trial.set_user_attr("replacement", balanced['replacement'])
        trial.set_user_attr("n_nodes", balanced['n_nodes'])
        trial.set_user_attr("n_edges", balanced['n_edges'])
        trial.set_user_attr("pareto_front_size", result['pareto_front_size'])
        
        # Report for pruning
        trial.report(score, step=0)
        
        return score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_optimization(
    graph_path: str,
    n_trials: int = 50,
    n_generations: int = 50,
    max_batch_per_gpu: int = 8,
    output_dir: str = "./optuna_results",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        graph_path: Path to the graph file (.pt)
        n_trials: Number of Optuna trials to run
        n_generations: Number of EA generations per trial (fixed for speed)
        max_batch_per_gpu: Maximum batch size per GPU
        output_dir: Directory to save results
        study_name: Name for the Optuna study (for persistence)
        storage: Database URL for distributed optimization (optional)
        
    Returns:
        Completed Optuna study
    """
    # Load graph
    logger.info(f"Loading graph from {graph_path}")
    graph = load_graph_data(Path(graph_path))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Base hyperparameters (fixed during optimization)
    base_hp = {
        "n_generations": n_generations,
        "max_batch_per_gpu": max_batch_per_gpu,
    }
    
    # Create study
    if study_name is None:
        study_name = f"ea_hp_optimization_{Path(graph_path).stem}"
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    
    logger.info(f"Starting optimization with {n_trials} trials")
    logger.info(f"Study name: {study_name}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, graph, base_hp),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save results
    logger.info("Optimization complete!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save study results
    results_file = output_path / f"{study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save full study as pickle for later analysis
    import joblib
    study_file = output_path / f"{study_name}_study.pkl"
    joblib.dump(study, study_file)
    logger.info(f"Study object saved to {study_file}")
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
        axes[0, 0].set_title("Optimization History")
        
        # Plot 2: Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
        axes[0, 1].set_title("Parameter Importances")
        
        # Plot 3: Parallel coordinate plot (first 8 params)
        params_to_plot = list(study.best_params.keys())[:8]
        optuna.visualization.matplotlib.plot_parallel_coordinate(
            study, params=params_to_plot, ax=axes[1, 0]
        )
        axes[1, 0].set_title("Parallel Coordinate Plot")
        
        # Plot 4: Slice plot for top 2 important params
        try:
            importance = optuna.importance.get_param_importances(study)
            top_params = list(importance.keys())[:2]
            optuna.visualization.matplotlib.plot_slice(study, params=top_params, ax=axes[1, 1])
            axes[1, 1].set_title("Slice Plot (Top 2 Parameters)")
        except:
            axes[1, 1].text(0.5, 0.5, "Slice plot unavailable", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plot_file = output_path / f"{study_name}_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_file}")
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for evolutionary algorithm"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        required=True,
        help="Path to the graph file (.pt)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)",
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=50,
        help="Number of EA generations per trial (default: 50)",
    )
    parser.add_argument(
        "--max-batch-per-gpu",
        type=int,
        default=8,
        help="Maximum batch size per GPU (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./optuna_results",
        help="Directory to save results (default: ./optuna_results)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study (default: auto-generated)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for distributed optimization (e.g., sqlite:///optuna.db)",
    )
    
    args = parser.parse_args()
    
    # Run optimization
    study = run_optimization(
        graph_path=args.graph_path,
        n_trials=args.n_trials,
        n_generations=args.n_generations,
        max_batch_per_gpu=args.max_batch_per_gpu,
        output_dir=args.output_dir,
        study_name=args.study_name,
        storage=args.storage,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value:.4f}" if isinstance(value, float) else f"  {param}: {value}")
    print(f"\nBalanced Solution Metrics:")
    for attr, value in study.best_trial.user_attrs.items():
        print(f"  {attr}: {value:.4f}" if isinstance(value, float) else f"  {attr}: {value}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
