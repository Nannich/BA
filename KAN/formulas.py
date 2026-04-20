import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

def trigonometric_sim1_gene12(x, lineage, s=0.37):
    trunk_val = 3.4 * np.cos(-1.4 * np.sin(1.2 * x + 2.2) + 0.7 * np.sin(2.1 * x - 9.3) + 2.2) + 4.0
    lin1_val  = 3.4 * np.cos(-1.4 * np.sin(1.2 * x + 2.2) + 0.7 * np.sin(2.1 * s - 9.3) + 2.2) + 4.0
    lin2_val  = 3.4 * np.cos(-1.4 * np.sin(1.2 * s + 2.2) + 0.7 * np.sin(2.1 * x - 9.3) + 2.2) + 4.0
    
    if lineage == 0:
        return np.where(x < s, trunk_val, lin1_val)
    else:
        return np.where(x < s, trunk_val, lin2_val)
    
import numpy as np

def polynomial_sim1_gene12(x, lineage, s=0.37):
    trunk_val = -3.6 * (0.7 - x)**2 - 1.7 * (-0.9 * x - 1)**2 + 9.4
    lin1_val  = -3.6 * (0.7 - s)**2 - 1.7 * (-0.9 * x - 1)**2 + 9.4
    lin2_val  = -3.6 * (0.7 - x)**2 - 1.7 * (-0.9 * s - 1)**2 + 9.4
    
    if lineage == 0:
        return np.where(x < s, trunk_val, lin1_val)
    else:
        return np.where(x < s, trunk_val, lin2_val)

def exponential_sim1_gene12(x, lineage, s=0.37):
    trunk_val = 12.6 - 5.4 * np.exp(0.6 * x)
    
    if lineage == 0:
        return trunk_val
    else:
        lin2_val = 12.6 - 5.4 * np.exp(0.6 * s)
        return np.where(x < s, trunk_val, lin2_val)
    

def pysr_pykan_sim1_gene12(x, lineage):
    if lineage == 0:  
        return 9.3604 - 7.212 * x - 7.212 * sigmoid(-6.2211 * x)
    else:
        return 5.6076 + sigmoid(4.0908 * x - 1.3232)


def pysr_mlp_sim1_gene12(x, lineage):
    if lineage == 0:  
        return np.exp(-3.1123 * (x**2) + 1.3772 * x + 1.284) + 2.0569   
    else:             
        return -3.163 * (x**3) + 3.7008 * (x**2) + 5.8862