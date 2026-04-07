from efficient_kan import KAN

HIDDEN_LAYERS = [16, 64, 64, 64, 64] 
GRID_SIZE = 5
SPLINE_ORDER = 3

def build_kan(input_dim, output_dim):
    layers = [input_dim] + HIDDEN_LAYERS + [output_dim]
    return KAN(layers, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER)