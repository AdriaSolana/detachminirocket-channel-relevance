### IMPORTS AND SETUP
from tsai.imports import warnings
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from detach_rocket.utils import feature_detachment, select_optimal_model, retrain_optimal_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy, rv_discrete
from scipy.linalg import LinAlgWarning
from collections import Counter
from aeon.datasets import load_classification
import copy
import matplotlib.pyplot as plt
import random

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



### UTILS
def download_dataset(dataset_name, truncate=None):
  X_train, y_train = load_classification(dataset_name, split="train")
  X_test, y_test = load_classification(dataset_name, split="test")

  if truncate:
      stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=truncate, random_state=seed)
      _, sample_indices = next(stratified_splitter.split(X_train, y_train))
      X_train, y_train = X_train[sample_indices], y_train[sample_indices]

      stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=truncate, random_state=seed)
      _, sample_indices = next(stratified_splitter.split(X_test, y_test))
      X_test, y_test = X_test[sample_indices], y_test[sample_indices]

  return X_train, X_test, y_train, y_test



### CLASSES
class MiniRocketFeatures(nn.Module):
    """This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza

    MiniRocket paper citation:
    @article{dempster_etal_2020,
      author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
      title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
      year    = {2020},
      journal = {arXiv:2012.08791}
    }
    Original paper: https://arxiv.org/abs/2012.08791
    Original code:  https://github.com/angus924/minirocket"""

    kernel_size, num_kernels, fitting = 9, 84, False

    def __init__(self, num_features=10_000, max_dilations_per_kernel=32, random_state=None):
        super(MiniRocketFeatures, self).__init__()
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel

    def fit(self, X, chunksize=128):
        self.c_in, self.seq_len = X.shape[1], X.shape[2]
        self.num_features = self.num_features // self.num_kernels * self.num_kernels # Multiple of 84 closest to num_features: 9996 for 10000

        # Convolution
        indices = torch.combinations(torch.arange(self.kernel_size), 3).unsqueeze(1) # Possible combinations of 3 elements (beta) within the 9 length kernel (so 84)
        kernels = (-torch.ones(self.num_kernels, 1, self.kernel_size)).scatter_(2, indices, 2) # (84, 1, 9) Fills the 84 kernels with -1 (alpha). Fills with 2 (beta) the combinations of 3 in 9.
        self.kernels = nn.Parameter(kernels.repeat(self.c_in, 1, 1), requires_grad=False) # (84 * num channels, 1, 9) Repeats the kernels for all the channels, redimensions and passes as parameter.

        # Dilations & padding
        self._set_dilations(self.seq_len) # Computes dilations according to seq_length, computes number of features per dilation

        # Channel combinations (multivariate)
        if self.c_in > 1:
            self._set_channel_combinations(self.c_in)

        # Bias
        for i in range(self.num_dilations):
            self.register_buffer(f'biases_{i}', torch.empty((self.num_kernels, self.num_features_per_dilation[i]))) # For each dilation, each kernel has features obtained by adding different biases.
        self.register_buffer('prefit', torch.BoolTensor([False])) # Biases are not initialized, only defined
        self.to(device) # Necessary to also fit with GPU

        num_samples = X.shape[0]
        if chunksize is None:
            chunksize = min(num_samples, self.num_dilations * self.num_kernels) # Deterministic
        else:
            chunksize = min(num_samples, chunksize) # Stochastic for chunksize < num_samples
        # np.random.seed(self.random_state)
        idxs = np.random.choice(num_samples, chunksize, False) # Choose samples to convolute and get the bias in the forward
        self.fitting = True
        if isinstance(X, np.ndarray):
            self(torch.from_numpy(X[idxs]).float().to(self.kernels.device))
        else:
            self(X[idxs].to(self.kernels.device))
        self.set_parameter_indices()
        self.fitting = False

        return self

    def forward(self, x):
        _features = []
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)): # Max 32
            _padding1 = i%2

            # Convolution: self.kernels is shape (84 * num channels, 1, 9), x is (batch size, num channels, seq length)
            C = F.conv1d(x, self.kernels, padding=padding, dilation=dilation, groups=self.c_in) # (batch size, 84 * num channels, sequence length) Each kernel convolutes each channel, giving seq length
            if self.c_in > 1: # multivariate
                C = C.reshape(x.shape[0], self.c_in, self.num_kernels, -1) # (batch size, num channels, num kernels, sequence length)
                channel_combination = getattr(self, f'channel_combinations_{i}') # (1, num channels, num kernels, 1)
                C = torch.mul(C, channel_combination) # (batch size, num channels, num kernels, sequence length) -> zeroes out some channels
                C = C.sum(1) # (batch size, num kernels, sequence length) -> sums channels
                # if not self.fitting: print(C[0])

            # Bias: if it has to be fitted yet
            if not self.prefit or self.fitting:
                num_features_this_dilation = self.num_features_per_dilation[i]
                bias_this_dilation = self._get_bias(C, num_features_this_dilation) # (num kernels, num features this dilation)
                setattr(self, f'biases_{i}', bias_this_dilation)
                if self.fitting:
                    if i < self.num_dilations - 1:
                        continue
                    else:
                        self.prefit = torch.BoolTensor([True])
                        return
                elif i == self.num_dilations - 1:
                    self.prefit = torch.BoolTensor([True])
            # Else for normal forward transform
            else:
                bias_this_dilation = getattr(self, f'biases_{i}') # (num kernels, num quantiles)

            # Features: append PPV
            _features.append(self._get_PPVs(C[:, _padding1::2], bias_this_dilation[_padding1::2])) # Selects kernels 0 2 4 ... or 1 3 5 ... -> Gets PPVs of half the kernels + dilation
            _features.append(self._get_PPVs(C[:, 1-_padding1::2, padding:-padding], bias_this_dilation[1-_padding1::2])) # Selects kernels 1 3 5 ... or kernels 0 2 4 ... and time pad:-pad -> Gets PPVs of the rest of the kernels - dilation
        return torch.cat(_features, dim=1)

    def set_parameter_indices(self):
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)): # Max 32
            # print('Dilation', i)
            _padding1 = i%2

            # Indices for the kernels / channel combinations & biases
            bias_this_dilation = getattr(self, f'biases_{i}')
            num_kernels, num_quantiles = bias_this_dilation.shape
            bias_indices = torch.arange(num_kernels*num_quantiles, dtype=int).reshape(num_quantiles, num_kernels).transpose(1, 0) # (num kernels, num quantiles)

            kernel_indices = torch.arange(num_kernels, dtype=int)
            kernel_indices = torch.stack([kernel_indices]*num_quantiles, dim=-1) # (num kernels, num quantiles)

            # "Features" for the even kernels.
            C_even = kernel_indices[_padding1::2] # (num kernels / 2, num quantiles)
            bias_this_dilation_even = bias_indices[_padding1::2] # (num kernels / 2, num quantiles)
            num_half_kernels = C_even.shape[0]

            # Simulate the PPV reshape
            C_even = C_even.flatten() # replaces .mean(2).flatten(1) and removes placeholder dimensions
            bias_this_dilation_even = bias_this_dilation_even.flatten()

            # Do the same for odd kernels
            C_odd = kernel_indices[1-_padding1::2] # (num kernels / 2, num quantiles)
            bias_this_dilation_odd = bias_indices[1-_padding1::2] # (num kernels / 2, num quantiles)

            # Simulate the PPV reshape
            C_odd = C_odd.flatten() # replaces .mean(2).flatten(1) and removes placeholder dimensions
            bias_this_dilation_odd = bias_this_dilation_odd.flatten()

            # Stack into flat arrays. Avoid reshaping because it messes up the order.
            C_full = torch.cat((C_even, C_odd))
            bias_this_dilation_full = torch.cat((bias_this_dilation_even, bias_this_dilation_odd))

            setattr(self, f'kernel_indices_{i}', C_full)
            setattr(self, f'bias_indices_{i}', bias_this_dilation_full)

        return


    def get_kernel_features(self, which, where):
        full_features = np.empty(shape=(0,), dtype=float)

        if which == 'channels':
            full_features = np.empty(shape=(0, self.c_in), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.c_in, axis=1)
        elif which == 'weights':
            full_features = np.empty(shape=(0, self.kernel_size), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.kernel_size, axis=1)

        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):

            biases_this_dilation = getattr(self, f'biases_{i}')
            num_quantiles = biases_this_dilation.shape[1]

            kernel_indices = getattr(self, f'kernel_indices_{i}')
            bias_indices = getattr(self, f'bias_indices_{i}')

            # Biases (=features): as many as num dilations * num kernels * num quantiles this dilation
            if which == 'biases':
                sorted_biases = biases_this_dilation.flatten()[bias_indices]
                full_features = np.append(full_features, sorted_biases.cpu().numpy()) # (num previous features + 84 * num quantiles)

            # Channel combinations: num_dilations * num_kernel, each kernel combines several channels
            elif which == 'channels':
                channel_combinations = getattr(self, f'channel_combinations_{i}')

                for q in range(0, num_quantiles):
                    selected_kernels = kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                    channel_combinations_q = channel_combinations[:, :, selected_kernels]
                    channel_combinations_q = torch.transpose(channel_combinations_q.squeeze(), 0, 1).cpu().numpy()

                    full_features = np.append(full_features, channel_combinations_q, axis=0) # (num previous features + 84 * num quantiles, num channels)

            # Weights: num dilations * num kernels, where each kernel has 9 weights
            elif which == 'weights':
                weights = self.kernels.view(-1, self.num_kernels, self.kernel_size)[0].cpu().numpy() # Kernels are equal for all channels. Does not work when setting the first dimension to n kernels

                for q in range(0, num_quantiles):
                    selected_kernels = kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                    weights_q = weights[selected_kernels]

                    full_features = np.append(full_features, weights_q, axis=0) # (num previous features + 84 * num quantiles, num channels)

            elif which == 'dilations':
                expanded_dilations =  np.repeat(dilation, self.num_kernels*num_quantiles, axis=0)
                full_features = np.append(full_features, expanded_dilations)

            elif which == 'paddings':
                expanded_dilations =  np.repeat(padding, self.num_kernels*num_quantiles, axis=0)
                full_features = np.append(full_features, expanded_dilations)

            else: raise ValueError(f'"{which}" is not recognized as a feature. Possible feaures are "biases", "channels", "weights", "dilations" or "paddings"')

        return np.where(where, full_features, np.nan)

    def _get_PPVs(self, C, bias):
        C = C.unsqueeze(-1) # (batch size, num kernels, seq length, 1)
        bias = bias.view(1, bias.shape[0], 1, bias.shape[1]) # (1, num kernels/2, 1, num quantiles)
        return (C > bias).float().mean(2).flatten(1) # Compute the average of the positive inequalites along the seq length axis. Flatten to (samples, num kernels/2*num quantiles) to obtain features

    def _set_dilations(self, input_length):
        num_features_per_kernel = self.num_features // self.num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dilations_per_kernel) # Limits the features per dilation per kernel to 32
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel
        max_exponent = np.log2((input_length - 1) / (9 - 1))
        dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True) # Return dilations exponentially spaced
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)
        remainder = num_features_per_kernel - num_features_per_dilation.sum()
        i = 0
        while remainder > 0: # Adjusts the number of features each dilation to match the desired number
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)
        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = len(dilations)
        self.dilations = dilations
        self.padding = []
        for i, dilation in enumerate(dilations):
            self.padding.append((((self.kernel_size - 1) * dilation) // 2))

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations # Possible kernels for all the dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1) # How many combinations of 2 channels can we get (max 10 -> max 1024 channels)
        # np.random.seed(self.random_state)
        num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(np.int32) # Each possible kernel*dilation has 1 channel to the max #combinations
        channel_combinations = torch.zeros((1, num_channels, num_combinations, 1))
        for i in range(num_combinations):
            channel_combinations[:, np.random.choice(num_channels, num_channels_per_combination[i], False), i] = 1 # From all the channels, set to 1 those that will be combined without repeating
        channel_combinations = torch.split(channel_combinations, self.num_kernels, 2) # split by dilation (Thanks!) Channel combinations is now splitted by kernel, so each kernel combines certain channels (indexed 1)
        for i, channel_combination in enumerate(channel_combinations):
            self.register_buffer(f'channel_combinations_{i}', channel_combination) # per dilation

    def _get_quantiles(self, n):
        return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).float() # select quantiles ranging from 0 to 1 according to the golden ratio * n mod 1

    def _get_bias(self, C, num_features_this_dilation):
        # Gets as many biases as features this dilation, from the quantiles of random samples
        # np.random.seed(self.random_state)
        idxs = np.random.choice(C.shape[0], self.num_kernels)
        samples = C[idxs].diagonal().T
        biases = torch.quantile(samples, self._get_quantiles(num_features_this_dilation).to(C.device), dim=1).T
        return biases

    def transform(self, o, normalize=False, chunksize=128):
        o = torch.tensor(o).float()
        if normalize: # No need to in minirocket
          mean = torch.mean(o, dim=2, keepdim=True)
          std = torch.std(o, dim=2, keepdim=True)
          o = (o - mean) / std

        return get_minirocket_features(o, self, chunksize)

def get_minirocket_features(o, model, chunksize=128, to_np=True):
    model = model.to(device)
    if isinstance(o, np.ndarray): o = torch.from_numpy(o).to(device)
    _features = []
    for oi in torch.split(o, chunksize):
        _features.append(model(oi.to(device)))
    features = torch.cat(_features)
    if to_np: return features.cpu().numpy()
    else: return features.unsqueeze()


class DeROCKETTransformer(TransformerMixin):
    def __init__(self, model_type:str='torchminirocket', num_kernels:int=10000, feature_fraction_kept=0.1, acc_size_tradeoff_coef=0.1):
        self.model_type = model_type
        self.num_kernels = num_kernels

        '''
        if self.model_type == "rocket":
            self.rocket_model = Rocket(num_kernels=self.num_kernels)
        elif self.model_type == "minirocket":
            self.rocket_model = MiniRocket(num_kernels=self.num_kernels)
        elif self.model_type == "multirocket":
            self.rocket_model = MultiRocket(num_kernels=self.num_kernels)
        elif self.model_type == "torchrocket":
            self.rocket_model = RocketPytorch(num_kernels=self.num_kernels)
        elif self.model_type == "torchminirocket":
            self.rocket_model = MiniRocketFeatures(num_features=self.num_kernels)
        elif self.model_type == "torchmultirocket":
            self.rocket_model = MultiRocketFeaturesPlus()
        '''
        if self.model_type == "torchminirocket":
            self.rocket_model = MiniRocketFeatures(num_features=self.num_kernels)
        else:
            raise ValueError('This version of DeROCKETTransformer only accepts "torchminirocket" as its base model')

        self.scaler = StandardScaler(with_mean=True)
        self.classifier = None
        self.feature_fraction_kept = feature_fraction_kept
        self.sfd_classifier = None
        self.alpha = None
        self.selection_mask = None
        self.is_fitted = False
        self._percentage_vector = None
        self._sfd_curve = None
        self._feature_importance_matrix = None
        self.acc_size_tradeoff_coef = acc_size_tradeoff_coef
        self.max_index = None

    # Basic transformer methods for the full ROCKET -> Scaler -> SFD workflow.
    def fit(self, X, y, verbose=0):
        self.rocket_model.fit(X)
        X_transform = self.rocket_model.transform(X)
        X_scaled_transform = self.scaler.fit_transform(X_transform)

        full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        full_classifier.fit(X_scaled_transform, y)
        self.classifier = RidgeClassifier(alpha=full_classifier.alpha_)
        self.classifier.fit(X_scaled_transform, y)

        # Select the best SFD% through a validation dataset
        if self.feature_fraction_kept == 'auto':
          X_train, X_val, y_train, y_val = train_test_split(X_scaled_transform, y, test_size=0.33, stratify=y, random_state=42)
          self.classifier.fit(X_train, y_train)
          self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(copy.copy(self.classifier), X_train, X_val, y_train, y_val, verbose=verbose)
          self.max_index, max_percentage = select_optimal_model(self._percentage_vector, self._sfd_curve, self._sfd_curve[0], self.acc_size_tradeoff_coef, graphics=False)
          self.selection_mask = self._feature_importance_matrix[self.max_index]>0
          self.sfd_classifier, _ = retrain_optimal_model(self.selection_mask, X_scaled_transform, y, None, self.max_index, verbose)

        # Just reduce to obtain the desired fraction of features. Get SFD classifier and selection mask
        else:
          _ = self._get_feature_fraction(self.classifier, X_scaled_transform, y, fraction_kept=self.feature_fraction_kept, verbose=verbose)

        self.alpha = self.sfd_classifier.alpha
        self.is_fitted = True
        return self

    def transform(self, X, verbose=0):
        if not self.is_fitted: raise NotFittedError('DeROCKETTransformer has not ben fitted yet.')
        X_transform = self.rocket_model.transform(X)
        X_scaled_transform = self.scaler.transform(X_transform)
        return X_scaled_transform[:, self.selection_mask]

    def fit_transform(self, X, y, verbose=0):
        self.rocket_model.fit(X)
        X_transform = self.rocket_model.transform(X)
        X_scaled_transform = self.scaler.fit_transform(X_transform)

        full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        full_classifier.fit(X_scaled_transform, y)
        self.classifier = RidgeClassifier(alpha=full_classifier.alpha_)
        self.classifier.fit(X_scaled_transform, y)

        # Select the best SFD% through a validation dataset
        if self.feature_fraction_kept == 'auto':
          X_train, X_val, y_train, y_val = train_test_split(X_scaled_transform, y, test_size=0.33, stratify=y, random_state=42)
          self.classifier.fit(X_train, y_train)

          self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(self.classifier, X_train, X_val, y_train, y_val, verbose=verbose)
          max_index, max_percentage = select_optimal_model(self._percentage_vector, self._sfd_curve, self._sfd_curve[0], 0.1, graphics=False)
          self.selection_mask = self._feature_importance_matrix[max_index]>0

          self.sfd_classifier, _ = retrain_optimal_model(self.selection_mask, X_scaled_transform, y, None, max_index, verbose)
          X_rerocket = X_scaled_transform[:, self.selection_mask]

        # Just reduce to obtain the desired fraction of features
        else:
          X_rerocket = self._get_feature_fraction(self.classifier, X_scaled_transform, y, fraction_kept=self.feature_fraction_kept, verbose=verbose)

        self.alpha = self.sfd_classifier.alpha
        self.is_fitted = True
        return X_rerocket

    # Basic transformer methods for just the ROCKET -> Scaler workflow.
    # Fits rocket model, scaler and ridge classifier
    def fit_rocket_classifier(self, X, y):
        _ = self._fit_full_rocket_model(X, y)
        return self

    # Returns scaled ROCKET transform before subsampling (SFD)
    def transform_rocket_classifier(self, X):
        X_transform = self.rocket_model.transform(X)
        return self.scaler.transform(X_transform)

    # Returns scaled ROCKET transform before subsampling (SFD)
    def fit_transform_rocket_classifier(self, X, y):
        return self._fit_full_rocket_model(X, y)

    # Basic transformer methods for just the SFD process.
    # Fits SFD to get a selection mask with a fraction_kept of features
    def fit_SFD(self, X_scaled_transform, y, fraction_kept):
        _ = self._get_feature_fraction(self.classifier, X_scaled_transform, y, fraction_kept=fraction_kept)
        return self

    def transform_SFD(self, X_scaled_transform):
        return X_scaled_transform[:, self.selection_mask]

    def fit_transform_SFD(self, X_scaled_transform, y, fraction_kept):
        return self.fit_SFD(X_scaled_transform, y, fraction_kept).transform_SFD(X_scaled_transform)

    # Inner methods and utilities
    def _fit_full_rocket_model(self, X, y, verbose=0):
        # Fit rocket model
        self.rocket_model.fit(X)
        X_transform = self.rocket_model.transform(X)

        # Scale
        scaler = StandardScaler()
        X_scaled_transform = scaler.fit_transform(X_transform)
        self.scaler = scaler

        # CV to find best alpha
        cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        cv_classifier.fit(X_scaled_transform, y)
        best_alpha_full = cv_classifier.alpha_
        if verbose >= 2:
            print('Best Alpha:', best_alpha_full)

        # Refit with all training set
        classifier = RidgeClassifier(alpha=best_alpha_full)
        classifier.fit(X_scaled_transform, y)
        self.classifier = classifier
        if verbose >= 1:
            print('Fit Accuracy before SFD: ', f'{classifier.score(X_scaled_transform, y):.2f}')

        return X_scaled_transform

    def _get_feature_fraction(self, classifier, X:np.ndarray, y:np.ndarray,
                              fraction_kept:float=0.1, drop_percentage:float=0.05, verbose=0):

        # fraction_kept = (1 - drop_percentage)^total_number_steps
        # +1 added to get closer to fraction_kept, not super relevant
        total_number_steps = (np.log(fraction_kept) / np.log(1 - drop_percentage)) + 1

        # Alpha and feature importance from full model
        aplha_value = classifier.alpha
        feature_importance_full = np.abs(classifier.coef_)[0,:]

        # Define percentage vector
        keep_percentage = 1-drop_percentage
        powers_vector = np.arange(total_number_steps)
        percentage_vector = np.power(keep_percentage, powers_vector)

        # Define lists and matrices
        feature_importance = np.copy(feature_importance_full)

        # Begin iterative feature selection
        for count, per in enumerate(percentage_vector):

            # Cumpute mask for selected features
            drop_percentage = 1 - per
            limit_value = np.quantile(feature_importance,drop_percentage)
            selection_mask = feature_importance >= limit_value

            # Apply mask
            X_subsampled = X[:,selection_mask]

            # Train model for selected features
            step_classifier = RidgeClassifier(alpha=aplha_value)
            step_classifier.fit(X_subsampled, y)

            # Kill masked features
            feature_importance[~selection_mask] = 0
            feature_importance[selection_mask] = np.abs(step_classifier.coef_)[0,:]

            if verbose >= 2:
                print("Step {} out of {}".format(count+1, total_number_steps))
                print('{:.3f}% of features used'.format(100*per))

        if verbose >= 1:
          print(f'Fit Accuracy after SFD to {fraction_kept*100}%: {step_classifier.score(X_subsampled, y):.2f}')

        self.sfd_classifier = step_classifier
        self.selection_mask = selection_mask
        return X_subsampled

    def _get_sfd_classifier(self):
        if not self.sfd_classifier:
            raise ValueError('DeROCKET does not have a SFD classifier. Use "transform" before this method.')
        return self.sfd_classifier

    def get_masked_kernel_features(self, which, mask=None):
        if self.model_type not in ('torchrocket', 'torchminirocket'): raise ValueError('This method is only available for the pytorch implementation of MiniROCKET and ROCKET (only for which == "channels")')
        if mask is None: mask = self.selection_mask
        return self.rocket_model.get_kernel_features(which, mask)

    def get_feature_importance_vector(self):
        if self.feature_fraction_kept != 'auto': raise ValueError('This method is only compatible with Detach-ROCKET (feature_fraction_kept = "auto")')
        return self._feature_importance_matrix[self.max_index]


class ReROCKETTransformer(TransformerMixin):
    def __init__(self, model_type:str='rocket', num_kernels:int=10000, num_models=10, feature_fraction_kept=0.1, acc_size_tradeoff_coef=0.1, random_state=None):
        self.model_type = model_type
        self.num_kernels = num_kernels
        self.num_models = num_models
        self.is_fitted = False
        self.derockets = [DeROCKETTransformer(model_type, num_kernels, feature_fraction_kept, acc_size_tradeoff_coef) for _ in range(num_models)]

    def fit(self, X, y, verbose=0):
        if(self.num_models > len(X)): raise ValueError('ReROCKET has more than one model for each X split')
        [self.derockets[i].fit(X[i], y[i], verbose=verbose) for i in range(self.num_models)]
        self.is_fitted = True
        return self

    def transform(self, X, verbose=0):
        if(self.num_models > len(X)): raise ValueError('ReROCKET has more than one model for each X split')
        if not self.is_fitted: raise NotFittedError('DeROCKETTransformer has not been fitted yet.')
        return [self.derockets[i].transform(X[i], verbose=verbose) for i in range(self.num_models)]

    def fit_transform(self, X, y, verbose=0):
        if(self.num_models > len(X)): raise ValueError('ReROCKET has more than one model for each X split')
        X_scaled_transform = [self.derockets[i].fit_transform(X[i], y[i], verbose=verbose) for i in range(self.num_models)]
        self.is_fitted = True
        return X_scaled_transform


    # ROCKET + SCALER
    def fit_rocket_classifiers(self, X, y):
        return [self.derockets[i].fit_rocket_classifier(X[i], y[i]) for i in range(len(X))]

    def transform_rocket_classifiers(self, X):
        return [self.derockets[i].transform_rocket_classifier(X[i]) for i in range(len(X))]

    def fit_transform_rocket_classifiers(self, X, y):
        return [self.derockets[i].fit_transform_rocket_classifier(X[i], y[i]) for i in range(len(X))]


    # SFD
    def fit_SFD(self, X, y, fraction_kept):
        self.is_fitted = True
        return [self.derockets[i].fit_SFD(X[i], y[i], fraction_kept) for i in range(len(X))]

    def transform_SFD(self, X):
        return [self.derockets[i].transform_SFD(X[i]) for i in range(len(X))]

    def fit_transform_SFD(self, X, y, fraction_kept):
        self.is_fitted = True
        return [self.derockets[i].fit_transform_SFD(X[i], y[i], fraction_kept) for i in range(len(X))]


    # UTILITIES
    def derocket_score(self, X, y):
        if not self.is_fitted: raise NotFittedError('DeROCKETTransformer has not been fitted yet.')
        return [self.derockets[i].sfd_classifier.score(X[i], y[i]) for i in range(len(y))]

    def get_classifiers(self):
      if not self.is_fitted: raise NotFittedError('DeROCKETTransformer has not been fitted yet.')
      return [d._get_sfd_classifier() for d in self.derockets]

    def get_full_classifiers(self):
      if not self.is_fitted: raise NotFittedError('DeROCKETTransformer has not been fitted yet.')
      return [d.classifier for d in self.derockets]

    def get_derocket_sizes(self):
      return [d._get_sfd_classifier().n_features_in_ for d in self.derockets]

    def get_alphas(self):
      return [d.alpha for d in self.derockets]

    def get_derocket_selection_masks(self):
      return np.array([d.selection_mask for d in self.derockets])

    def get_masked_kernel_features(self, which, mask=None):
      return np.array([d.get_masked_kernel_features(which, mask) for d in self.derockets])

    def get_feature_importances(self):
      return np.array([d.get_feature_importance_vector() for d in self.derockets])


    # CHANNEL SELECTION
    def show_channel_p_value(self, n_runs=1000):
      num_total_features = 84 * (self.num_kernels // 84)
      channel_combinations_orig = self.get_masked_kernel_features(which='channels', mask=np.ones(num_total_features)) # (num models, num features, num channels)

      channel_combinations_derocket = self.get_masked_kernel_features(which='channels') # (num models, num features, num channels)
      channel_distribution = np.nanmean(channel_combinations_derocket, axis=(0, 1)) # (num channels)
      derocket_entropy = entropy(channel_distribution)

      channels_nan_mask = np.isnan(channel_combinations_derocket)
      num_selected_features = np.sum(~np.any(channels_nan_mask, axis=-1))

      entropy_distribution = []
      for run in range(n_runs):
        random_features = np.random.choice(num_total_features, size=num_selected_features, replace=False)
        channel_combination_random = channel_combinations_orig[:, random_features]
        channel_distribution_random = np.mean(channel_combination_random, axis=(0, 1))
        entropy_distribution.append(entropy(channel_distribution_random))

      plt.figure()
      bins = np.arange(-0.01, np.max(entropy_distribution) + 0.01, 0.02)
      entropy_count, _, _ = plt.hist(entropy_distribution, bins=bins)
      plt.vlines(derocket_entropy, ymin=0, ymax=max(entropy_count), colors='orange')
      plt.show()

      entropy_probs = entropy_count / entropy_count.sum()
      dist_indices = list(range(len(entropy_probs)))
      entropy_discrete_dist = rv_discrete(values=(dist_indices, entropy_probs))

      derocket_entropy_quantile = np.max(dist_indices)
      for i in dist_indices[:-1]:
        if (bins[i] <= derocket_entropy and derocket_entropy < bins[i+1]):
          derocket_entropy_quantile = i
          break

      p_value = entropy_discrete_dist.cdf(derocket_entropy_quantile)
      print('P-value', p_value)

      plt.figure()
      plt.plot(entropy_discrete_dist.cdf(dist_indices))
      plt.vlines(derocket_entropy_quantile, ymin=0, ymax=1, colors='orange')
      plt.show()

      return


    # PLOT
    def plot_channel_frequency(self):
      if self.model_type not in ('torchrocket', 'torchminirocket'): raise ValueError('This method is only available for the pytorch implementation of MiniROCKET and ROCKET (only for which == "channels")')
      rocket_features = 84 * (self.num_kernels // 84) if self.model_type == 'torchminirocket' else self.num_kernels*2
      channel_combinations_orig = self.get_masked_kernel_features(which='channels', mask=np.ones(rocket_features)) # (num models, num features, num channels)
      channel_combinations_derocket = self.get_masked_kernel_features(which='channels') # (num models, num features, num channels)

      plt.figure(figsize=(16, 16))

      ##  COMBINATION SIZE HISTOGRAMS
      # MiniROCKET combinations
      plt.subplot(2, 2, 1)
      ones_count_per_row = np.nansum(np.vstack(channel_combinations_orig), axis=1)
      max_ones_per_row = np.max(ones_count_per_row)
      bin_edges = np.arange(0.5, max_ones_per_row + 1.5, 1)
      plt.hist(ones_count_per_row, bins=bin_edges, edgecolor='black')
      plt.xticks(np.arange(1, max_ones_per_row + 1, 1))
      plt.xlabel('Number of channel group sizes')
      plt.ylabel('Frequency')
      plt.title(f'Channel grouping sizes in MiniROCKET w/ {rocket_features*self.num_models} kernels', fontsize=12)

      # Detach-ROCKET combinations
      plt.subplot(2, 2, 2)
      channels_nan_mask = np.isnan(channel_combinations_derocket)
      active_kernels_count = np.sum(~np.any(channels_nan_mask, axis=-1)) # "Rows" (channels) that do not have any nan (=active kernels)
      ones_count_per_row = np.nansum(np.vstack(channel_combinations_derocket), axis=1)
      plt.hist(ones_count_per_row, bins=bin_edges, edgecolor='black')
      plt.xticks(np.arange(1, max_ones_per_row + 1, 1))
      plt.xlabel('Number of channel group sizes')
      plt.ylabel('Frequency')
      plt.title(f'Channel grouping sizes in Detach-ROCKET w/ {active_kernels_count} kernels ({active_kernels_count/(rocket_features*self.num_models)*100:.2f}% kept)', fontsize=12)

      ##  CHANNEL FREQUENCIES HISTOGRAMS
      # Channel occurence for kernels with just 1 channel
      plt.subplot(2, 2, 3)
      num_channels_in_kernel = np.sum(channel_combinations_derocket == 1, axis=-1)
      channel_indices = np.nanargmax(channel_combinations_derocket[num_channels_in_kernel == 1], axis=-1) + 1
      bin_edges = np.arange(0.5, channel_combinations_derocket.shape[-1] + 1.5, 1)
      plt.hist(channel_indices, bins=bin_edges, edgecolor='black')
      plt.xlabel('Channel')
      plt.ylabel('Frequency')
      plt.title(f'Histogram of single channels in kernels', fontsize=12)

      # Chanel occurrence relative to the number of kernels
      plt.subplot(2, 2, 4)
      percentage_rows_with_one = np.nanmean(channel_combinations_derocket, axis=(0, 1)) * 100
      plt.bar(np.arange(1, channel_combinations_derocket.shape[-1] + 1), percentage_rows_with_one, edgecolor='black')
      plt.xlabel('Channel')
      plt.ylabel('Frequency')
      plt.title(f'Histogram of channel occurence relative to the number of kernels', fontsize=12)

      return


class ReROCKETPredictor(BaseEstimator, ABC):
    def __init__(self, classifier_type, ridge_classifiers):
        self.classifier_type = classifier_type
        self.ridge_classifiers = ridge_classifiers
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit_predict(self, X, y):
        pass

    def set_ridge_classifiers(self, classifiers):
        self.ridge_classifiers = classifiers


class LinearReROCKETPredictor(ReROCKETPredictor):
    def __init__(self, ridge_classifiers):
        super().__init__('linear', ridge_classifiers)

    def fit(self, X, y):
        # Select the most restrictive alpha and set the classifier
        alphas = [c.alpha for c in self.ridge_classifiers[:len(X)]]
        voted_alpha = max(alphas)
        classifier = RidgeClassifier(alpha=voted_alpha)

        # Reshape X and fit the classifier
        X = np.hstack(X)
        classifier.fit(X, y)
        self.linear_classifier = classifier
        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted: raise NotFittedError('ReROCKETPredictor has not been fitted yet.')
        X = np.hstack(X)
        return self.linear_classifier.predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class SoftVoteReROCKETPredictor(ReROCKETPredictor):
    def __init__(self, classifier_type, ridge_classifiers):
        super().__init__(classifier_type, ridge_classifiers)

    def vote(self, X):
        votes = []

        for i, X_single in enumerate(X):
          classifier = self.ridge_classifiers[i]
          votes.append(classifier.decision_function(X_single))

        return np.stack(votes, axis=-1) # (sample, rep)


class SoftVoteAvgReROCKETPredictor(SoftVoteReROCKETPredictor):
    def __init__(self, ridge_classifiers):
        super().__init__('softvoteavg', ridge_classifiers)
        self.is_fitted = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        if not self.is_fitted: raise NotFittedError('ReROCKETPredictor has not been fitted yet.')
        votes = self.vote(X)
        votes = np.average(votes, axis=-1)
        return np.where(votes >= 0, self.ridge_classifiers[0].classes_[1], self.ridge_classifiers[0].classes_[0])

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class SoftVoteTrainReROCKETPredictor(SoftVoteReROCKETPredictor):
    def __init__(self, ridge_classifiers):
        super().__init__('softvotetrain', ridge_classifiers)

    def fit(self, X, y):
        votes = self.vote(X)
        scaler = StandardScaler(with_mean=False)
        votes = scaler.fit_transform(votes)
        self.scaler = scaler

        soft_classifier = LogisticRegressionCV(Cs=np.logspace(-6, 1, 6))
        soft_classifier.fit(votes, y)
        self.soft_classifier = soft_classifier
        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted: raise NotFittedError('ReROCKETPredictor has not been fitted yet.')
        votes = self.vote(X)
        votes = self.scaler.transform(votes)
        return self.soft_classifier.predict(votes)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class HardVotingReROCKETPredictor(ReROCKETPredictor):
    def __init__(self, classifier_type, ridge_classifiers):
          super().__init__(classifier_type, ridge_classifiers)

    def vote(self, X):
      votes = []

      for i, X_single in enumerate(X):
        classifier = self.ridge_classifiers[i]
        votes.append(classifier.predict(X_single))

      return np.stack(votes, axis=-1) # (sample, rep)


class HardVoteReROCKETPredictor(HardVotingReROCKETPredictor):
    def __init__(self, ridge_classifiers):
        super().__init__('hardvote', ridge_classifiers)
        self.is_fitted = True

    def fit(self, X, y):
        pass

    def predict(self, X):
        if not self.is_fitted: raise NotFittedError('ReROCKETPredictor has not been fitted yet.')
        votes = self.vote(X)
        hard_votes = [Counter(v).most_common()[0][0] for v in votes]

        # Convert to integers if needed
        if isinstance(self.ridge_classifiers[0].classes_[0], int):
            hard_votes = [int(hv) for hv in hard_votes]

        return hard_votes

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class HardVoteTrainReROCKETPredictor(HardVotingReROCKETPredictor):
    def __init__(self, ridge_classifiers):
        super().__init__('hardvotetrain', ridge_classifiers)

    def fit(self, X, y):
        votes = self.vote(X)
        categories = [self.ridge_classifiers[0].classes_]*votes.shape[1]
        enc = OrdinalEncoder(categories=categories, dtype=int)
        votes = enc.fit_transform(votes)
        self.encoder = enc

        soft_classifier = LogisticRegressionCV(Cs=np.logspace(-6, 1, 6))
        soft_classifier.fit(votes, y)
        self.soft_classifier = soft_classifier
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted: raise NotFittedError('ReROCKETPredictor has not been fitted yet.')
        votes = self.vote(X)
        votes = self.encoder.transform(votes)
        return self.soft_classifier.predict(votes)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

