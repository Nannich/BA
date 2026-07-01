import re
import sympy
import numpy as np
import pandas as pd
from pathlib import Path

class sigmoid(sympy.Function):
    nargs = 1
    def _eval_derivative(self, symbol):
        return self.func(self.args[0]) * (1 - self.func(self.args[0]))
        
    def _eval_evalf(self, prec):
        arg_val = float(self.args[0].evalf(prec))
        import math
        return sympy.Float(1 / (1 + math.exp(-arg_val)), prec)


def evaluate_equations(eq_csv_path, ground_truth_path, eval_out_dir, file_token):
    """
    Calculates derivative for each gene to determine whether it is an activator or
    repressor. Evaluates it against the ground truth.
    """
    eq_csv_path = Path(eq_csv_path)
    ground_truth_path = Path(ground_truth_path)
    eval_out_dir = Path(eval_out_dir)
    
    if not eq_csv_path.exists() or not ground_truth_path.exists():
        print("Missing equations or ground truth file references.")
        return

    # Load ground truth file
    gt_df = pd.read_csv(ground_truth_path)
    gt_df.columns = [col.capitalize() for col in gt_df.columns]
    
    # Standardize ground truth keys to lowercase and stripped of extra spaces
    gt_edges = {
        (str(row['Gene1']).lower().strip(), str(row['Gene2']).lower().strip()): str(row['Type']).strip() 
        for _, row in gt_df.iterrows()
    }

    # Load math expressions file
    df_equations = pd.read_csv(eq_csv_path)
    summary_records = []
    
    # Track unique ground-truth edges recovered by the equations
    recovered_gt_edges = set()

    for _, row in df_equations.iterrows():
        target_gene = str(row["TargetGene"]).strip()
        expr_string = str(row["Equation"]).strip()
        
        # Inject * to pass the equation on to sympy
        expr_string = re.sub(r'([\d\)])([a-zA-Z])', r'\1*\2', expr_string)
        
        try:
            local_dict = {"sigmoid": sigmoid}
            parsed_expr = sympy.parse_expr(expr_string, local_dict=local_dict)
            predictor_symbols = [s for s in parsed_expr.free_symbols if s.name != 'True']
            
            for symbol in predictor_symbols:
                pred_gene = symbol.name
                
                # Form normalized tracking keys for network lookup
                pred_key = pred_gene.lower().strip()
                target_key = target_gene.lower().strip()
                
                if (pred_key, target_key) not in gt_edges:
                    continue
                    
                # Mark biological edge as successfully recovered/retained
                recovered_gt_edges.add((pred_key, target_key))
                
                true_sign = gt_edges[(pred_key, target_key)]
                
                free_vars = list(parsed_expr.free_symbols)
                modules_dict = {"sigmoid": lambda x: 1 / (1 + np.exp(-x))}
                fast_numpy_func = sympy.lambdify(free_vars, parsed_expr, modules=[modules_dict, "numpy"])

                grid_size = 50
                sampled_points = np.linspace(0.0, 2.0, grid_size)

                input_matrix = []
                for var in free_vars:
                    if var == symbol:
                        input_matrix.append(sampled_points)
                    else:
                        input_matrix.append(np.ones(grid_size))

                grid_outputs = fast_numpy_func(*input_matrix)
                global_slope, _ = np.polyfit(sampled_points, grid_outputs, 1)

                pred_sign = "+" if global_slope > 0 else "-"
                status_str = "CORRECT" if pred_sign == true_sign else "WRONG"
                
                summary_records.append({
                    "TargetGene": target_gene,
                    "PredictorGene": pred_gene,
                    "TrueSign": true_sign,
                    "PredictedSign": pred_sign,
                    "GlobalSlope": float(global_slope),
                    "Status": status_str
                })
                
        except Exception as e:
            print(f"Error gene: {target_gene}: {e}")
            continue

    # Export detailed sign evaluation matrix
    df_eval_details = pd.DataFrame(summary_records)
    csv_out_path = eval_out_dir / f"{file_token}_eval.csv"
    df_eval_details.to_csv(csv_out_path, index=False)
    print(f"Saved evaluation spreadsheet to: {csv_out_path}")

    # Calculate Topology Metrics
    total_gt_count = len(gt_edges)
    retained_gt_count = len(recovered_gt_edges)
    incorrectly_removed_count = total_gt_count - retained_gt_count
    
    prop_incorrectly_removed = (incorrectly_removed_count / total_gt_count) if total_gt_count > 0 else 0.0
    prop_retained = (retained_gt_count / total_gt_count) if total_gt_count > 0 else 0.0

    # Calculate Functional Metrics
    correct_signs = sum(1 for r in summary_records if r["Status"] == "CORRECT")
    total_evaluated_links = len(summary_records)
    sign_accuracy = (correct_signs / total_evaluated_links) if total_evaluated_links > 0 else 0.0

    summary_metrics = {
        "total_ground_truth_edges": total_gt_count,
        "retained_edges_count": retained_gt_count,
        "incorrectly_removed_count": incorrectly_removed_count,
        "prop_edges_retained": round(float(prop_retained), 4),
        "prop_edges_incorrectly_removed": round(float(prop_incorrectly_removed), 4),
        "retained_edges_evaluated": total_evaluated_links,
        "correct_signs_count": correct_signs,
        "directional_sign_accuracy": round(float(sign_accuracy), 4)
    }
    
    df_summary = pd.DataFrame([summary_metrics])
    summary_csv_path = eval_out_dir / f"{file_token}_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved machine-readable performance summary to: {summary_csv_path}")