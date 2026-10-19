import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws) if Ws is not None else None
    first_layer = np.maximum(0,np.dot(x, W1))
    second_layer=np.maximum(0,np.dot(first_layer, W2))
    third_layer=np.maximum(0,np.dot(second_layer,W3)+(np.dot(x, Ws) if Ws is not None else x))
    return np.round(third_layer, decimals=4).tolist()