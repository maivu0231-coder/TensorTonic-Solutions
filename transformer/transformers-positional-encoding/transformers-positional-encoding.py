import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position =np.arange(seq_length)[:,np.newaxis]
    i=np.arange(d_model)[np.newaxis,:]
    angle_rates=1/np.power(10000,(2*(i//2)/np.float64(d_model)))
    angle_rads=position*angle_rates
    return np.where(i%2==0,np.sin(angle_rads),np.cos(angle_rads)).astype(np.float64)