def eli_grad(signal):
    thr = 3*1e-3
    grad_signal = grad_sig(signal)
    grad_index = np.where(abs(grad_signal) > thr)
    return signal[grad_index]
  
def grad_sig(signal):
    return np.diff(signal)
