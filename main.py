import argparse

from src.core.config import DATA_DIR, MODELS_DIR, FIGURES_DIR, ensure_dir, set_global_seed
from src.core.utils import resolve_gene_identifier
from src.core.preprocessing import run_preprocessing

from src.trajectory.train_trajectory import run_trajectory
from src.trajectory.plot_trajectory import run_plotting as run_plot_trajectory
from src.trajectory.eval_trajectory import run_eval as run_eval_trajectory

from src.grn.extract_grn import run_grn as run_extract_grn
from src.grn.plot_grn import run_plot_grn
from src.grn.benchmark_grn import run_benchmark

from src.symbolic.extract_symbolic import run_symbolic_pipeline
from src.symbolic.plot_symbolic import run_plot_symbolic


def build_parser():
    parser = argparse.ArgumentParser()
    domain_subparsers = parser.add_subparsers(dest="domain", required=True)

    base_dataset_parser = argparse.ArgumentParser(add_help=False)
    base_dataset_parser.add_argument("dataset", type=str, help="Target dataset directory identifier.")

    # Trajectory
    parser_traj = domain_subparsers.add_parser("trajectory", help="Trajectory modeling commands.")
    traj_actions = parser_traj.add_subparsers(dest="action", required=True)

    traj_train = traj_actions.add_parser("train", parents=[base_dataset_parser])
    traj_train.add_argument("--gene", type=str, required=True, help="Target gene identifier or 'all'.")
    traj_train.add_argument("--loss", type=str, choices=["mse", "zinb"], default="zinb")
    traj_train.add_argument("--model", type=str, choices=["kan", "mlp", "null"], default="kan")

    traj_eval = traj_actions.add_parser("eval", parents=[base_dataset_parser])
    
    traj_plot = traj_actions.add_parser("plot", parents=[base_dataset_parser])
    traj_plot.add_argument("--mode", type=str, required=True, choices=["scatter", "distribution", "trajectory", "cluster"])
    traj_plot.add_argument("--gene", type=str, default=None, help="Target gene identifier or 'all'.")
    traj_plot.add_argument("--model", type=str, default="kan")
    traj_plot.add_argument("--loss", type=str, default="mse")

    # GRN
    parser_grn = domain_subparsers.add_parser("grn", help="Gene Regulatory Network commands.")
    grn_actions = parser_grn.add_subparsers(dest="action", required=True)

    grn_extract = grn_actions.add_parser("extract", parents=[base_dataset_parser])
    grn_extract.add_argument("--input_mode", type=str, choices=["log", "smooth"], default="log")
    grn_extract.add_argument("--target_mode", type=str, choices=["log", "smooth"], default="log")
    grn_extract.add_argument("--loss", type=str, choices=["mse", "zinb"], default="mse")
    grn_extract.add_argument("--lag", type=float, default=0.0)
    grn_extract.add_argument("--ground_truth", action="store_true")

    grn_plot = grn_actions.add_parser("plot", parents=[base_dataset_parser])
    grn_plot.add_argument("--sensitivity", type=float, default=0.15)
    grn_plot.add_argument("--ground_truth", action="store_true")
    grn_plot.add_argument("--experiment_name", type=str, default=None)
    grn_plot.add_argument("--arch", type=str, default=None)
    grn_plot.add_argument("--input_mode", type=str, default="log")
    grn_plot.add_argument("--target_mode", type=str, default="log")
    grn_plot.add_argument("--lag", type=float, default=0.0)
    grn_plot.add_argument("--deep", action="store_true")

    # Symbolic
    parser_sym = domain_subparsers.add_parser("symbolic", help="Symbolic equation extraction.")
    sym_actions = parser_sym.add_subparsers(dest="action", required=True)

    sym_extract = sym_actions.add_parser("extract", parents=[base_dataset_parser])
    sym_extract.add_argument("--mode", type=str, default="grn", choices=["grn", "trajectory"])
    sym_extract.add_argument("--arch", type=str, default="smo_log_l")

    sym_plot = sym_actions.add_parser("plot", parents=[base_dataset_parser])
    sym_plot.add_argument("--checkpoint", type=str, required=True, help="Direct path to the target .pth KAN checkpoint file.")

    # Becnhmark
    parser_benchmark = domain_subparsers.add_parser("benchmark", help="Run GRN benchmark.")
    parser_benchmark.add_argument("search_path", type=str, help="Root directory to discover datasets.")

    # Preprocessing
    parser_process = domain_subparsers.add_parser("process", parents=[base_dataset_parser], help="Cache dataset.")

    return parser


def main():
    set_global_seed(1)
    parser = build_parser()
    args = parser.parse_args()

    if args.domain == "benchmark":
        run_benchmark(args)
        return

    ensure_dir(DATA_DIR)

    if args.domain == "process":
        run_preprocessing(dataset_name=args.dataset)
        return

    dataset = run_preprocessing(dataset_name=args.dataset)    

    if args.domain in ["trajectory", "grn", "symbolic"]:
        if args.domain in ["grn", "symbolic"]:
            sub_path = args.domain
        else:
            sub_path = getattr(args, "arch", None) or getattr(args, "experiment_name", None) or args.domain
            
        args.model_dir = ensure_dir(MODELS_DIR / args.dataset / sub_path)
        args.fig_dir = ensure_dir(FIGURES_DIR / args.dataset / sub_path)

    if hasattr(args, "gene") and args.gene not in (None, "all"):
        args.gene = resolve_gene_identifier(args.gene, dataset)

    target_genes = list(range(len(dataset.gene_names))) if getattr(args, "gene", None) == "all" else [getattr(args, "gene", None)]

    if args.domain == "trajectory":
        if args.action == "train":
            for gene in target_genes:
                args.gene = gene
                run_trajectory(args, dataset)
        elif args.action == "plot":
            for gene in target_genes:
                args.gene = gene
                run_plot_trajectory(args, dataset)
        elif args.action == "eval":
            run_eval_trajectory(args, dataset)

    elif args.domain == "grn":
        if args.action == "extract":
            run_extract_grn(args, dataset)
        elif args.action == "plot":
            run_plot_grn(args)

    elif args.domain == "symbolic":
        if args.action == "extract":
            run_symbolic_pipeline(dataset_name=args.dataset, mode=args.mode, arch_name=args.arch)
        elif args.action == "plot":
            run_plot_symbolic(args)


if __name__ == "__main__":
    main()