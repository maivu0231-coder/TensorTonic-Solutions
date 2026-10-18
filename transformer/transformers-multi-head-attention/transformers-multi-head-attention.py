import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    Q=np.dot(Q, W_q)
    K=np.dot(K, W_k)
    V=np.dot(V, W_v)
    batch_size,seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    Q=Q.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)
    K=K.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)
    V=V.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)
    scores=np.matmul(Q,K.transpose(0,1,3,2))/np.sqrt(d_k)
    scores=scores-np.max(scores,axis=-1,keepdims=True)
    attention_weights=np.exp(scores)/np.sum(np.exp(scores),axis=-1,keepdims=True)
    attention_output=np.matmul(attention_weights,V)
    attention_output=attention_output.transpose(0,2,1,3).reshape(batch_size,seq_len,d_model)
    output=np.dot(attention_output,W_o)
    return output
