import numpy as np
import torch
from scipy import signal

def define_velocity_fourier(sample_amps, ntimepoint, phase, voffset):
    famps = sample_amps * np.exp(1j * phase)
    velocity = np.fft.irfft(famps, ntimepoint)
    velocity += voffset
    return velocity

def add_baseline_period(t, x, baseline_duration, baseline_value=0.0):    
    dt = t[1] - t[0]
    npoints_baseline = int(np.ceil(baseline_duration/dt))
    x_with_baseline = np.concatenate((baseline_value*np.ones(npoints_baseline), x))
    t_with_baseline = np.linspace(0, t.max() + baseline_duration, np.size(x_with_baseline))
    return t_with_baseline, x_with_baseline


def upsample(y_input, n, tr):
    if y_input.ndim == 1:
        y_input = np.expand_dims(y_input, 0).T
    npoints, ncols = np.shape(y_input)
    y_interp = np.zeros((n, ncols))
    x = tr * np.arange(npoints)
    for icol in range(ncols):
        y = y_input[:, icol]
        xvals = np.linspace(np.min(x), np.max(x), n)
        y_interp[:, icol] = np.interp(xvals, x, y)
    return y_interp

def input_batched_signal_into_NN_area(s_data_for_nn, NN_model, xarea, area):
    ntime = s_data_for_nn.shape[0]
    num_slice_to_use = s_data_for_nn.shape[1]
    nfeature = num_slice_to_use + 2 # area takes up last two features
    feature_length = xarea.size
    nwindows = ntime // feature_length
    remainder = ntime % feature_length
    
    velocity_NN = np.zeros((nwindows + (1 if remainder > 0 else 0)) * feature_length)
    
    def run_window(s_window, start_idx):
        x_np = np.zeros((1, nfeature, feature_length))
        for islice in range(num_slice_to_use):
            x_np[0, islice, :] = s_window[:, islice].squeeze()
        x_np[0, -2, :] = xarea
        x_np[0, -1, :] = area
        
        device = next(NN_model.parameters()).device
        x = torch.from_numpy(x_np).float().to(device)
        y_predicted_tensor = NN_model(x)
        y_predicted = y_predicted_tensor.detach().cpu().numpy().squeeze()
        velocity_NN[start_idx:start_idx + feature_length] = y_predicted

    for w in range(nwindows):
        ind1, ind2 = w * feature_length, (w + 1) * feature_length
        run_window(s_data_for_nn[ind1:ind2], w * feature_length)

    if remainder > 0:
        leftover = s_data_for_nn[-remainder:]
        pad_len = feature_length - remainder
        padded = np.pad(leftover, ((0, pad_len), (0, 0)), mode="reflect")
        run_window(padded, nwindows * feature_length)

        velocity_NN = velocity_NN[:ntime]

    return velocity_NN

def scale_data(s):
    raw_mean = np.mean(s, axis=0)
    detrended = signal.detrend(s, axis=0)
    return detrended / raw_mean