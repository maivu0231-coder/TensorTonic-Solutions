
import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    Ws = np.array(Ws, dtype=float)

    # Main path: Linear + ReLU
    first_layer = np.maximum(0, np.dot(x, W1))

    # Main path + shortcut, then ReLU
    second_layer = np.maximum(
        0,
        np.dot(first_layer, W2) + np.dot(x, Ws)
    )

    return np.round(second_layer, 4).tolist()
