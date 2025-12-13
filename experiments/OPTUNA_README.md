# Hyperparameter Optimization for Evolutionary Algorithm

This directory contains tools for optimizing the hyperparameters of the NSGA-II evolutionary algorithm using Optuna.

## Files

- **`optuna_hp_search.py`**: Command-line script for running hyperparameter optimization
- **`optuna_hp_optimization.ipynb`**: Interactive Jupyter notebook for hyperparameter optimization and analysis
- **`models.py`**: Contains the `EAHyperparameters` dataclass with all tunable parameters

## Hyperparameters

The following hyperparameters are optimized:

### Algorithm Parameters
- `population_size`: Size of the population (20-100)
- `mutation_rate`: Probability of mutation (0.05-0.5)
- `crossover_rate`: Probability of crossover (0.5-0.95)
- `mutation_sigma`: Standard deviation for Gaussian mutation (0.05-0.3)

### Objective Weights
- `w_completeness`: Weight for completeness in quality objective (0.1-2.0)
- `w_replacement`: Weight for replacement in quality objective (0.1-2.0)
- `w_complexity_node`: Weight for log(nodes) in complexity objective (1.0-15.0)
- `w_complexity_edge`: Weight for log(edges) in complexity objective (0.5-10.0)

## Usage

### Command Line

```bash
# Basic usage
python optuna_hp_search.py --graph-path graphs/war.pt --n-trials 50

# With custom settings
python optuna_hp_search.py \
    --graph-path graphs/war.pt \
    --n-trials 100 \
    --n-generations 30 \
    --max-batch-per-gpu 8 \
    --output-dir ./optuna_results \
    --study-name war_optimization

# With persistent storage (for distributed optimization)
python optuna_hp_search.py \
    --graph-path graphs/war.pt \
    --n-trials 100 \
    --storage sqlite:///optuna_study.db
```

### Jupyter Notebook

Open `optuna_hp_optimization.ipynb` and run the cells sequentially. The notebook provides:
- Interactive hyperparameter optimization
- Detailed visualizations
- Analysis of optimization results
- Testing of best hyperparameters

## Optimization Objective

The optimization maximizes a composite score:

```
score = quality - 0.1 * complexity
```

Where:
- **Quality** = `w_completeness * completeness + w_replacement * replacement` (higher is better)
- **Complexity** = `w_complexity_node * log(nodes) + w_complexity_edge * log(edges)` (lower is better)

The score is computed on the "balanced" solution from the Pareto front (knee point approximation).

## Output

The optimization produces:

1. **JSON results file** (`<study_name>_results.json`):
   - Best trial number
   - Best score
   - Best hyperparameters
   - Metrics of the balanced solution

2. **Study pickle file** (`<study_name>_study.pkl`):
   - Full Optuna study object for later analysis

3. **Visualization plots** (`<study_name>_plots.png`):
   - Optimization history
   - Parameter importances
   - Parallel coordinate plot
   - Slice plot

## Example Output

```
OPTIMIZATION COMPLETE
================================================================================

Best Trial: #23
Best Score: 1.8542

Best Hyperparameters:
  population_size: 60
  mutation_rate: 0.1523
  crossover_rate: 0.8234
  mutation_sigma: 0.1245
  w_completeness: 0.6543
  w_replacement: 1.2345
  w_complexity_node: 9.8765
  w_complexity_edge: 4.5678

Balanced Solution Metrics:
  quality: 2.1234
  complexity: 2.6920
  completeness: 0.8765
  replacement: 0.9234
  n_nodes: 45
  n_edges: 123
  pareto_front_size: 18

================================================================================
```

## Advanced Usage

### Distributed Optimization

Run multiple workers in parallel using a shared database:

```bash
# Worker 1
python optuna_hp_search.py \
    --graph-path graphs/war.pt \
    --n-trials 50 \
    --storage sqlite:///shared_study.db \
    --study-name war_distributed

# Worker 2 (on another machine/GPU)
python optuna_hp_search.py \
    --graph-path graphs/war.pt \
    --n-trials 50 \
    --storage sqlite:///shared_study.db \
    --study-name war_distributed
```

### Custom Objective Function

Modify the `objective()` function in `optuna_hp_search.py` to optimize for different metrics:

```python
# Example: Optimize for minimum complexity while maintaining quality > 0.8
def objective(trial, graph, base_hp):
    hp = EAHyperparameters(...)
    result = run_ea_optimization(graph=graph, hp=hp)
    
    quality = result['balanced']['quality']
    complexity = result['balanced']['complexity']
    
    if quality < 0.8:
        return float('inf')  # Constraint violation
    
    return complexity  # Minimize complexity
```

## Tips

1. **Start with fewer trials** (20-30) to get initial insights
2. **Adjust n_generations** based on your time budget (lower = faster trials)
3. **Use distributed optimization** for large search spaces
4. **Analyze parameter importances** to focus on key hyperparameters
5. **Run multiple studies** with different objective functions to explore trade-offs

## Dependencies

```bash
pip install optuna matplotlib plotly joblib
```

All other dependencies are part of the circuit-tracer project.
