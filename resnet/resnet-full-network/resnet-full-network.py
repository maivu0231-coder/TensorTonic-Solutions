
import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.float64)

    # 1. Initial projection
    a = np.maximum(0, x @ conv1)

    # 2. Identity block
    h = np.maximum(0, a @ W1_b1)
    h = h @ W2_b1
    a = np.maximum(0, h + a)

    # 3. Projection block
    h = np.maximum(0, a @ W1_b2)
    h = h @ W2_b2
    shortcut = a @ Ws_b2
    a = np.maximum(0, h + shortcut)

    # 4. Final linear classifier
    logits = a @ fc

    return np.round(logits, 4).tolist()
