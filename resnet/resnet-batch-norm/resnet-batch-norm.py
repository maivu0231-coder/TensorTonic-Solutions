import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    def batch_norm(x,gamma,beta):
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        x_hat = (x - mean) / np.sqrt(variance + 1e-5)
        return gamma * x_hat + beta
    def relu(x):
        return np.maximum(0, x)
    def residual_block(x,W1,W2,gamma1,beta1,gamma2,beta2,mode):
        x=np.array(x)
        if mode=="post":
            out1=np.dot(x,W1)
            out1=batch_norm(out1,gamma1,beta1)
            out1=relu(out1)
            out2=np.dot(out1,W2)
            out2=batch_norm(out2,gamma2,beta2)
            out=out2+x
            out=relu(out)
            return out
        elif mode=="pre":
            out1=batch_norm(x,gamma1,beta1)
            out1=relu(out1)
            out2=np.dot(out1,W1)
            out2=batch_norm(out2,gamma2,beta2)
            out2=relu(out2)
            out=np.dot(out2,W2)
            out=out+x
            return out
    output=residual_block(x,W1,W2,gamma1,beta1,gamma2,beta2,mode)
    return {"output": np.round(output, 4).tolist(), "mode": mode}