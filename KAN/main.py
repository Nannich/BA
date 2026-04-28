import os
import argparse
from train import run_training
from visualize import run_visualization
from symbolic import run_extraction
from de import run_de

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sim", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="paul")
    parser.add_argument("--data_dir", type=str, default="~/BA/data/")
    parser.add_argument("--model_dir", type=str, default="~/BA/models/")
    parser.add_argument("--fig_dir", type=str, default="~/BA/figs/")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--model", type=str, choices=["effkan", "pykan", "mlp", "null"], default="effkan")
    parser_train.add_argument("--gene", type=int, default=None) # Choose None to train all

    # Visualize command
    parser_vis = subparsers.add_parser("visualize")
    parser_vis.add_argument("name", type=str)
    parser_vis.add_argument("gene", type=int)

    # Symbolic command
    parser_sym = subparsers.add_parser("symbolic")
    parser_sym.add_argument("name", type=str)
    parser_sym.add_argument("gene", type=int)
    
    # Differential expression tests command
    parser_sym = subparsers.add_parser("de")
    parser_sym.add_argument("name", type=str)
    parser_sym.add_argument("--lineage", type=int, default=0)

    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.model_dir = os.path.expanduser(args.model_dir)
    args.fig_dir = os.path.expanduser(args.fig_dir)



    # Run the correct script based on the command
    if args.command == "train":
        run_training(args)
    elif args.command == "visualize":
        run_visualization(args)
    elif args.command == "symbolic":
        run_extraction(args)
    elif args.command == "de":
        run_de(args)


if __name__ == "__main__":
    main()


# Interesting Genes
# Bifurcating 1:        0, 12, 300, 3000
# Multifurcating 9:     18, 23, 26
# Paul: