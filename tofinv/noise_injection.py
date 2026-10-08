
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np
import pickle

def add_gaussian_noise(X, nslice, mean=0, gauss_low=0.01, gauss_high=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if not 0 < nslice <= X.shape[1] - 2 or not 0 <= gauss_low <= gauss_high:
        raise ValueError("Invalid slice count or Gaussian noise bounds.")
    std = rng.uniform(low=gauss_low, high=gauss_high, size=(X.shape[0], 1, 1))
    noise = rng.normal(mean, 1.0, (X.shape[0], nslice, X.shape[2])) * std
    X[:, :nslice, :] += noise
    return X

def add_pca_noise(X, model, nslice, scalemax=1.5, rng=None):
    noise = sample_noise(X, model, nslice=nslice, rng=rng)
    noise_scaled = scale_noise(X, noise, nslice=nslice, scalemax=scalemax, rng=rng)
    X[:, :nslice, :] += noise_scaled
    return X

def define_pca_model(noise_data, n_component=25):
    noise_data = np.asarray(noise_data, dtype=float)
    if noise_data.ndim != 2 or noise_data.shape[0] < 2:
        raise ValueError("Noise bank must contain at least two time-series samples.")
    n_component = min(n_component, noise_data.shape[0], noise_data.shape[1])
    pca = PCA(n_components=n_component)
    coeffs = pca.fit_transform(noise_data) 
    kde_models = []
    for index in range(n_component):
        coefficient = coeffs[:, index]
        kde_models.append(None if np.ptp(coefficient) < 1e-12 else gaussian_kde(coefficient))
    return {'pca': pca, 'kdes': kde_models, 'coefficient_means': coeffs.mean(axis=0)}

def sample_noise(X, model, nslice, rng=None):
    nsample, ntime = X.shape[0], X.shape[-1]
    pca = model['pca']
    kdes = model['kdes']
    ncomp = len(kdes)
    rng = np.random.default_rng() if rng is None else rng
    coefficients = []
    for index, kde in enumerate(kdes):
        if kde is None:
            coefficients.append(np.full(nslice * nsample, model['coefficient_means'][index]))
        else:
            seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            coefficients.append(kde.resample(nslice * nsample, seed=seed).ravel())
    coeffs = np.vstack(coefficients).T
    noise = coeffs @ pca.components_
    new_shape = (nsample, nslice, ntime)
    noise_reshaped = np.reshape(noise, new_shape)
    return noise_reshaped

def scale_noise(X, noise, nslice, scalemax=1.0, scaleoverride=None, rng=None):
    noise_scaled = noise.copy()
    slc1_max = X[:, 0, :].max(axis=1, keepdims=True)
    sample_peaks = np.abs(noise_scaled).max(axis=-1, keepdims=True) + 1e-9
    noise_scaled = noise_scaled / sample_peaks
    if scaleoverride:
        scale = scaleoverride * np.ones((X.shape[0], 1, 1))
    else:
        rng = np.random.default_rng() if rng is None else rng
        scale = rng.uniform(0, scalemax, size=(X.shape[0], 1, 1))
    noise_scaled *= (scale * slc1_max[:, None, :])
    return noise_scaled

def save_pca_model(model, pca_model_path):
    with open(pca_model_path, "wb") as f:
        pickle.dump(model, f)

def load_pca_model(pca_model_path):
    with open(pca_model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_noise_data(noise_data_path):
    with open(noise_data_path, 'rb') as f:
        noise = pickle.load(f)
    return noise
