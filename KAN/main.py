import ctypes
import os

try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import os
import argparse
from train import run_training
from visualize import run_visualization
from symbolic import run_extraction
from de import run_de
from grn import run_grn
from preprocessing import run_preprocessing

def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model_dir", type=str, default="./models/")
    parser.add_argument("--fig_dir", type=str, default="./figures/")

    # Parse dataset
    parser.add_argument("--dataset", type=str, default="paul", choices=["paul", "gsd", "gsd50", "hsc", "hsc70", "bf"])

    # Parse command specific arguments
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--model", type=str, choices=["effkan", "pykan", "mlp", "null"], default="effkan")
    parser_train.add_argument("--gene", type=int, default=None) # Choose None to train all

    parser_vis = subparsers.add_parser("visualize")
    parser_vis.add_argument("name", type=str)
    parser_vis.add_argument("gene", type=int)

    parser_sym = subparsers.add_parser("symbolic")
    parser_sym.add_argument("name", type=str)
    parser_sym.add_argument("gene", type=int)
    
    parser_sym = subparsers.add_parser("de")
    parser_sym.add_argument("name", type=str)
    parser_sym.add_argument("--lineage", type=int, default=0)

    parser_sym = subparsers.add_parser("grn")
    parser_sym.add_argument("name", type=str)

    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.model_dir = os.path.expanduser(args.model_dir)
    args.fig_dir = os.path.expanduser(args.fig_dir)

    # Fetch and preprocess the dataset
    adata, pseudotime, weights = run_preprocessing(args)    

    # Run the correct script based on the command
    if args.command == "train":
        run_training(args, adata, pseudotime, weights)
    elif args.command == "visualize":
        run_visualization(args, adata, pseudotime, weights)
    elif args.command == "symbolic":
        run_extraction(args, adata, pseudotime, weights)
    elif args.command == "de":
        run_de(args, adata, pseudotime, weights)
    elif args.command == "grn":
        run_grn(args, adata, pseudotime, weights)


if __name__ == "__main__":
    main()