def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    first_layer=np.maximum(0, np.dot(x, W1.T))
    second_layer=np.maximum(0,np.dot(first_layer, W2.T)+x)
    return second_layer.tolist()