import numpy as np
def increasing_linear_array(k):
    # Create linearly increasing values from 0 to 1
    assert k >= 1 and k == int(k)
    values = np.linspace(1/k, 1, k)    
    if k == 1:
        return np.array([1])
    # Normalize to sum to 1
    return values / np.sum(values)

print(increasing_linear_array(30)) 