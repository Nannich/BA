import torch
from efficient_kan import KAN as EFFKAN
from kan import KAN as PYKAN

N_GENES = 8


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def print_diagnostics(model):
    for name, param in model.named_parameters():
        print(f"{name:<30} {str(list(param.shape)):<15} {param.numel()}")


def main():
    data_dim = N_GENES - 1
    print(f"N_GENES: {N_GENES}\n")

    # scKAN
    eff_kan_shape = [data_dim, data_dim*2+1, (data_dim*2+1)*2+1, data_dim*2+1, 1]
    model_effkan = EFFKAN(eff_kan_shape, grid_size=10, spline_order=3)
    effkan_single = get_n_params(model_effkan)
    effkan_total = effkan_single * N_GENES

    print("scKAN")
    print(f"widths: {eff_kan_shape}")
    print(f"p_single: {effkan_single}")
    print(f"p_total: {effkan_total}")
    print_diagnostics(model_effkan)
    print()

    # lagKAN
    model_pykan = PYKAN(width=[data_dim, 1], grid=3, k=3, auto_save=False)
    pykan_single = get_n_params(model_pykan)
    pykan_total = pykan_single * N_GENES

    print("lagKAN")
    print(f"widths: [{data_dim}, 1]")
    print(f"p_single: {pykan_single}")
    print(f"p_total: {pykan_total}")
    print_diagnostics(model_pykan)


if __name__ == "__main__":
    main()