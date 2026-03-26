
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np
import pickle


def add_gaussian_noise(X, nslice=3, mean=0, gauss_low=0.01, gauss_high=0.1):
    gauss_noise_std = np.random.uniform(low=gauss_low, high=gauss_high)
    noise = np.random.normal(mean, gauss_noise_std, (nslice, X.shape[2]))
    X[:, :nslice, :] += noise
    return X


def add_pca_noise(X, model, nslice=3, scalemax=1.5):
    noise = sample_noise(X, model)
    noise_scaled = scale_noise(X, noise, scalemax=scalemax)
    X[:, :nslice, :] += noise_scaled
    
    # X[:, :nslice, :] += sample_noise(X, model)
    return X


def define_pca_model(noise_data, n_component=25):
    pca = PCA(n_components=n_component)
    coeffs = pca.fit_transform(noise_data) 
    kde_models = [gaussian_kde(coeffs[:, i]) for i in range(n_component)]
    return {'pca': pca, 'kdes': kde_models}


def sample_noise(X, model, nslice=3):
    nsample, ntime = X.shape[0], X.shape[-1]
    pca = model['pca']
    kdes = model['kdes']
    ncomp = len(kdes)
    coeffs = np.vstack([kdes[i].resample(nslice*nsample) for i in range(ncomp)]).T
    noise = coeffs @ pca.components_
    new_shape = (nsample, nslice, ntime)
    noise_reshaped = np.reshape(noise, new_shape)
    return noise_reshaped


def scale_noise(X, noise, nslice=3, scalemax=1.0, scaleoverride=None):
    noise_scaled = noise.copy()
    slc1_max = X[:, 0, :].max(axis=1)
    scale = np.random.uniform(0, scalemax, size=X.shape[0])
    if scaleoverride:
        scale = scaleoverride
    scale_fact = scale * slc1_max
    for islice in range(nslice):
        noise_scaled[:, islice, :] /= noise_scaled[:, islice, :].max()
        noise_scaled[:, islice, :] *= scale_fact[:, None]
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


