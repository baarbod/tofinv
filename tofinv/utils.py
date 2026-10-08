import numpy as np
import torch
from scipy import signal

def available_cpu_count():
    try:
        return max(1, len(__import__("os").sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, (__import__("os").cpu_count() or 1))

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
    if s_data_for_nn.ndim != 2:
        raise ValueError("Input signal must have shape (timepoints, slices).")
    ntime = s_data_for_nn.shape[0]
    num_slice_to_use = s_data_for_nn.shape[1]
    feature_length = xarea.size
    nwindows = ntime // feature_length
    remainder = ntime % feature_length
    
    if feature_length < 2:
        raise ValueError("Area profile must contain at least two timepoints.")
    if area.shape != xarea.shape:
        raise ValueError("Area and position profiles must have matching shapes.")
    velocity_NN = np.zeros((nwindows + (1 if remainder > 0 else 0)) * feature_length)
    
    def run_window(s_window, start_idx):
        flow_np = np.zeros((1, num_slice_to_use, feature_length))
        for islice in range(num_slice_to_use):
            flow_np[0, islice, :] = s_window[:, islice].squeeze()
            
        area_np = np.zeros((1, 1, feature_length))
        area_np[0, 0, :] = area
        
        device = next(NN_model.parameters()).device
        flow_x = torch.from_numpy(flow_np).float().to(device)
        area_x = torch.from_numpy(area_np).float().to(device)
        
        with torch.no_grad():
            y_predicted_tensor = NN_model(flow_x, area_x)
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
    s = np.asarray(s, dtype=float)
    if s.ndim not in (1, 2) or s.shape[0] < 2:
        raise ValueError("Signal scaling requires at least two timepoints.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Signal contains NaN or infinite values.")
    raw_mean = np.mean(s, axis=0)
    if np.any(np.abs(raw_mean) < 1e-8):
        raise ValueError("Signal contains a near-zero mean channel and cannot be normalized safely.")
    detrended = signal.detrend(s, axis=0)
    scaled = detrended / raw_mean
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Signal scaling produced non-finite values.")
    return scaled

def scale_area(xarea, area):
    xarea, area = np.asarray(xarea), np.asarray(area)
    if xarea.ndim != 1 or area.ndim != 1 or xarea.shape != area.shape:
        raise ValueError("Area and position profiles must be matching one-dimensional arrays.")
    middle_index = xarea.size // 2
    reference_area = area[middle_index]
    if not np.isfinite(reference_area) or reference_area <= 0:
        raise ValueError("Area profile midpoint must be finite and positive.")
    area_scaled = area / reference_area
    if not np.all(np.isfinite(area_scaled)) or np.any(area_scaled <= 0):
        raise ValueError("Area profile must contain finite positive values.")
    return area_scaled
