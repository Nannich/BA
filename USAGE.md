# Usage Examples

## Command Structure
```bash
python3 main.py <domain> <action> <dataset> [flags]
```

---

## 1. Data Preprocessing
Caches raw expression and pseudotime CSV matrices into a `.npz` file.

```bash
python3 main.py process dyn-LL-2000-5
```

* **File Storage Mapping:**
  * **Reads:** `data/raw/**/<dataset>/ExpressionData.csv` and `PseudoTime.csv`.
  * **Saves:** `data/processed/<dataset>.npz`.

---

## 2. Trajectory Domain

### Train Trajectory Model
Fits continuous trajectories for genes across lineages.

```bash
python3 main.py trajectory train dyn-LL-2000-5 --gene all --loss zinb --model kan --ridge_lambda 0.1
```
* `--gene`: Target gene identifier string, or `all` to run across the entire dataset.
* `--loss`: Optimization objective. Options: `mse`, `zinb` (default).
* `--model`: Network architecture baseline. Options: `kan` (default), `mlp`, `null`.
* `--ridge_lambda`: Ridge regularization coefficient applied to the ZINB loss function (default: 0.1).
* **File Storage Mapping:**
  * **Reads:** `data/processed/<dataset>.npz`.
  * **Saves:** `models/<dataset>/trajectory/<model>_<gene>_<loss>.pth`.

### Plot Trajectory Data
Visualizes scatter distributions, raw histograms, or model-fit spline trajectories.

```bash
python3 main.py trajectory plot dyn-LL-2000-5 --mode trajectory --gene g1 --model kan --loss mse
```
* `--mode`: Type of visual layout. Options: `scatter`, `distribution`, `trajectory`, `cluster`.
* `--gene`: Target gene identifier string, or `all`.
* `--model` / `--loss`: Used to find and match the correct checkpoint file.
* **File Storage Mapping:**
  * **Reads:** `data/processed/<dataset>.npz` and `models/<dataset>/trajectory/<model>_<gene>_<loss>.pth`.
  * **Saves:** `results/figures/<dataset>/trajectory/<dataset>_<mode>_<gene_or_model>_<loss>.png`.

### Evaluate Trajectory Models
Automatic benchmarking across all found trajectory checkpoints, saving a summary matrix.

```bash
python3 main.py trajectory eval dyn-LL-2000-5
```
* **File Storage Mapping:**
  * **Reads:** `data/processed/<dataset>.npz` and all matching `models/<dataset>/trajectory/*.pth` binaries.
  * **Saves:** `models/<dataset>/trajectory/<dataset>_trajectory_evaluation.csv`.

---

## 3. GRN Domain

### Extract GRN Edges
Infers GRNs; architecture can be customized.

```bash
python3 main.py grn extract dyn-LL-2000-5 --input_mode smooth --target_mode log --loss mse --lag 0.1
```
* `--input_mode`: Expression matrix type for predictors. Options: `log` (default), `smooth`.
* `--target_mode`: Expression matrix type for target genes. Options: `log` (default), `smooth`.
* `--loss`: Objective function optimization penalty. Options: `mse` (default), `zinb`.
* `--lag`: Relative time-lag offset.
* `--ground_truth`: Enable knowledge-based network masking.
* **File Storage Mapping:**
  * **Reads:** `data/processed/<dataset>.npz`. If smoothing is chosen, it also loads checkpoints from `models/<dataset>/trajectory/kan_<gene>_mse.pth`. If `--ground_truth` is active, it loads `data/raw/**/<dataset>/GroundTruthNetwork.csv`.
  * **Saves:** Training model weight configurations to `models/<dataset>/grn/<experiment_name>/<target_gene>_checkpoint.pth` and the final edge layout file to `results/grn/<dataset>/<experiment_name>/rankedEdges.csv`.

### Plot Inferred Graphs
Renders graph illustrating activation and repression edges between genes.

```bash
python3 main.py grn plot dyn-LL-2000-5 --sensitivity 0.15 --arch smo_log_l
```
* `--sensitivity`: Filters out edges below this weight threshold.
* `--arch`: Architecture sub-folder keyword used to find the `rankedEdges.csv` targets.
* `--experiment_name`: Direct architecture sub-folder keyword override.
* `--deep`: Identifies deep model architecture layouts.
* `--ground_truth`: To plot the ground truth network.
* **File Storage Mapping:**
  * **Reads:** `results/grn/<dataset>/<experiment_name_or_arch>/rankedEdges.csv` or `data/raw/**/<dataset>/GroundTruthNetwork.csv`.
  * **Saves:** Graph visualization to `results/figures/<dataset>/grn/<dataset>_<experiment_name>_network.png`.

---

## 4. Symbolic Formula Domain

### Extract Symbolic Equations
Batch processes weights to reconstruct algebraic functional representations.

```bash
python3 main.py symbolic extract dyn-LL-2000-5 --arch log_log_l --prune
```
* `--arch`: Architecture identifier keyword matching the target GRN checkpoints folder (default: `log_log_l`).
* `--skip_deep`: Extract expressions directly from the original weights without running deep symbolic retraining.
* `--prune`: Applies structural pruning to remove minor connections and low-weight variables before parsing.
* **File Storage Mapping:**
  * **Reads:** Network weights at `models/<dataset>/grn/<arch>/*_checkpoint.pth` or `models/<dataset>/trajectory/kan_*_mse.pth`, ground truth at `data/raw/**/<dataset>/GroundTruthNetwork.csv`.
  * **Saves:**
    * Equation spreadsheets to `results/symbolic/<dataset>/formulas/<file_token>_equations.csv`.
    * Network graphs to `results/figures/<dataset>/symbolic/<file_token>/<gene_symbol>_symbolic_graph.png`.
    * Performance metrics to `results/symbolic/<dataset>/eval/<file_token>_eval.csv`.
    * Symbolic KAN checkpoints to `models/<dataset>/symbolic/deep_<checkpoint_name>.pth`.

### Plot Symbolic Networks
Plots the KAN graph with all splines.

```bash
python3 main.py symbolic plot dyn-LL-2000-5 --checkpoint models/dyn-LL-2000-5/symbolic/deep_g1_checkpoint.pth
```
* `--checkpoint`: Direct path to the target `.pth` KAN checkpoint file.
* **File Storage Mapping:**
  * **Reads:** Checkpoints at the provided `--checkpoint` path.
  * **Saves:** Plot to `results/symbolic/<dataset>/<checkpoint_file_stem>.png`.

---

## 5. System Benchmarking
Evaluates modeling performance across dataset subdirectories.

```bash
python3 main.py benchmark data/raw/BEELINE-data/inputs/Synthetic --mode grn
```
* `--mode`: Target domain validation focus. Options: `grn` (default), `trajectory`, `symbolic`.
* **File Storage Mapping:**
  * **Reads:** Scans the provided `<search_path>` parameter to locate folders hosting `ExpressionData.csv` and `PseudoTime.csv`, plus validation networks at `GroundTruthNetwork.csv`.
  * **Saves:** Comparison CSVs at `results/benchmark/<search_path_folder_name>/` containing accuracy matrices (e.g., `auroc_individual.csv`, `auprc_ratio_median.csv`, or task summary spreadsheets).
```