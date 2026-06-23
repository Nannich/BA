import torch
import matplotlib.pyplot as plt
from pathlib import Path
from kan import KAN as PyKAN
import numpy as np

from src.core.config import RESULTS_DIR, ensure_dir
from src.core.preprocessing import run_preprocessing

def plot_symbolic(checkpoint_path, dataset_obj=None, folder="./figures"):
    """Loads a KAN checkpoint, runs a forward pass, and plots it."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = {k.replace("kan.", ""): v for k, v in ckpt["state_dict"].items()}
    
    # Resolve architecture width dimensions
    width = ckpt.get("width")
    if not width:
        if "predictor_names" in ckpt:
            in_dim = len(ckpt["predictor_names"])
        elif dataset_obj:
            in_dim = dataset_obj.pseudotime.shape[1]
        else:
            in_dim = 1

        if ckpt.get("loss_mode", ckpt.get("loss")) == "zinb":
            out_dim = 3
        else:
            out_dim = 1
            
        width = [in_dim] + ckpt.get("hidden_layers", []) + [out_dim]

    # Instantiate PyKAN structure and load parameters
    model = PyKAN(width=width, grid=ckpt.get("grid", 3), k=ckpt.get("k", 3), device="cpu", auto_save=False)
    model.load_state_dict(sd)
    model.eval()

    if "predictor_names" in ckpt:
        in_vars = ckpt["predictor_names"]
    else:
        if isinstance(width[0], list):
            in_dim = width[0][0]
        else:
            in_dim = width[0]
        in_vars = [f"pt{i+1}" for i in range(in_dim)]

    tgt = ckpt.get("target_gene", ckpt.get("gene_names", ["gene"])[0])
    
    if isinstance(width[-1], list):
        final_out_dim = width[-1][0]
    else:
        final_out_dim = width[-1]

    if final_out_dim == 3:
        out_vars = [f"{tgt}_mu", f"{tgt}_theta", f"{tgt}_pi"]
    else:
        out_vars = [tgt]

    # Execute activation pass to populate edge weight metrics
    X = ckpt.get("X_numpy")
    if X is None:
        if dataset_obj:
            X = dataset_obj.pseudotime
        else:
            if isinstance(width[0], list):
                dummy_dim = width[0][0]
            else:
                dummy_dim = width[0]
            X = np.ones((100, dummy_dim), dtype=np.float32)

    with torch.no_grad():
        model(torch.tensor(X, dtype=torch.float32))

    model.plot(folder=str(folder), in_vars=in_vars, out_vars=out_vars, title=f"KAN: {tgt}")
    return plt.gcf()


def run_plot_symbolic(args):
    """Handles data dependencies and figure export."""
    file = Path(args.checkpoint)
    ckpt = torch.load(file, map_location="cpu", weights_only=False)
    
    if "predictor_names" in ckpt:
        data = None
    else:
        data = run_preprocessing(dataset_name=args.dataset)
        
    out_dir = ensure_dir(RESULTS_DIR / "symbolic" / args.dataset / file.parent.name)
    
    fig = plot_symbolic(file, data, out_dir)
    fig.savefig(out_dir / f"{file.stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved completed blueprint network diagram at: {out_dir / file.stem}.png")