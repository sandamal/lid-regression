import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from scipy import optimize
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler


def read_dataset_file(f):
    with open(f) as dataset:
        x = []
        y = []
        cols = dataset.readline().split(',')
        print(cols)

        global colmap
        colmap = {}
        for i, col in enumerate(cols):
            if ':' in col:
                if col.split(':')[0] in colmap:
                    colmap[col.split(':')[0]].append(i - 1)
                else:
                    colmap[col.split(':')[0]] = [i - 1]
        for line in dataset:
            line = [float(val) for val in line.split(',')]
            y.append(line[0])
            x.append(line[1:])

        return np.matrix(x), np.array(y).reshape(-1, )


# lid of a batch of query points X as well its neighbors
def mle_batch_neighbors(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=a)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    a = a[tuple(idx)]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a, sort_indices


def get_lids_with_neighbors(X, k=10):
    lid_batch, sort_indices = mle_batch_neighbors(X, X, k=k)
    lids = np.asarray(lid_batch, dtype=np.float32)
    neighbor_lids = lids[sort_indices]
    avg_neighbor_lids = np.mean(neighbor_lids, axis=1)
    lid_ratio = np.divide(avg_neighbor_lids, lids)
    return lids, lid_ratio


def get_kde(data):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-1, 1, 30)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3, n_jobs=-1)
    grid.fit(data.reshape(-1, 1))
    return grid.best_estimator_


def classification_perf(y, y_hat, scenario, cv, index, elapsed_time, column_names):
    mse = mean_squared_error(y, y_hat)
    return pd.DataFrame([[scenario, cv, index, mse, elapsed_time]], columns=column_names)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def weight_calculation(normal_lids, attack_lids, normal_density, attack_density, weight_lb):
    # density of normal lids from the normal distribution
    d_normal_normal = normal_density.score_samples(normal_lids.reshape(-1, 1))
    d_normal_normal = np.exp(d_normal_normal)
    # density of normal lids from the attack distribution
    d_normal_attack = attack_density.score_samples(normal_lids.reshape(-1, 1))
    d_normal_attack = np.exp(d_normal_attack)

    # to avoid div by 0
    zero_indices = np.where(d_normal_attack == 0)[0]
    d_normal_attack[zero_indices] = np.inf
    min_value = np.min(d_normal_attack)
    d_normal_attack[zero_indices] = min_value

    lr_normal = np.divide(d_normal_normal, d_normal_attack)
    # clip any lr above 25
    np.clip(lr_normal, 0, 25, out=lr_normal)
    lr_normal_shape = lr_normal.shape

    # density of fl lids from the normal distribution
    d_attack_normal = normal_density.score_samples(attack_lids.reshape(-1, 1))
    d_attack_normal = np.exp(d_attack_normal)
    # density of fl lids from the attack distribution
    d_attack_attack = attack_density.score_samples(attack_lids.reshape(-1, 1))
    d_attack_attack = np.exp(d_attack_attack)

    # to avoid div by 0
    zero_indices = np.where(d_attack_attack == 0)[0]
    d_attack_attack[zero_indices] = np.inf
    min_value = np.min(d_attack_attack)
    d_attack_attack[zero_indices] = min_value

    lr_attack = np.divide(d_attack_normal, d_attack_attack)
    np.clip(lr_attack, 0, 25, out=lr_attack)
    lr_fl_shape = lr_attack.shape

    tmp_weights = np.concatenate((lr_normal, lr_attack), axis=0).reshape(-1, 1)

    # scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(weight_lb, 1))
    scaler.fit(tmp_weights)

    lr_normal = scaler.transform(lr_normal.reshape(-1, 1))
    lr_normal.shape = lr_normal_shape
    lr_attack = scaler.transform(lr_attack.reshape(-1, 1))
    lr_attack.shape = lr_fl_shape

    return lr_normal, lr_attack


def tanh_func(x, a, b):
    return 0.5 + 0.5 * np.tanh(a * x - b)


def inv_tanh_func(x, a, b):
    return 0.5 - 0.5 * np.tanh(a * x - b)


def get_model(dataset, num_features):
    if 'house' == dataset:
        # create model
        model = Sequential()
        model.add(Dense(16, input_dim=num_features, kernel_initializer='normal', activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model, 100, 60
    elif 'loan' == dataset:
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=num_features, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model, 100, 40
    elif 'machine' == dataset:
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=num_features, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model, 100, 20
    elif 'omnet' == dataset:
        # create model
        model = Sequential()
        model.add(Dense(4, input_dim=num_features, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model, 100, 10


def clean_lid_ratios(cutoff, lid_ratio, normal_sample_size, weight_lb):
    indices = np.argwhere((lid_ratio < cutoff) & (lid_ratio != np.inf)).ravel()
    trunc_lid_ratio = lid_ratio[indices]
    normal_indices = indices[indices < normal_sample_size]
    attack_indices = indices[indices >= normal_sample_size]
    normal_lids_ratio = lid_ratio[normal_indices]
    attack_lids_ratio = lid_ratio[attack_indices]
    normal_density = get_kde(normal_lids_ratio)
    attack_density = get_kde(attack_lids_ratio)
    normal_lr, attacked_lr = weight_calculation(normal_lids_ratio, attack_lids_ratio, normal_density, attack_density,
                                                weight_lb)
    tmp_lr = np.concatenate((normal_lr, attacked_lr), axis=0)
    # fit a tanh function
    params, params_covariance = optimize.curve_fit(inv_tanh_func, trunc_lid_ratio, tmp_lr)
    # obtain the weights from the fitted function
    normal_weights = inv_tanh_func(normal_lids_ratio, params[0], params[1])
    attack_weights = inv_tanh_func(attack_lids_ratio, params[0], params[1])
    weights = np.vstack((normal_weights.reshape(-1, 1), attack_weights.reshape(-1, 1)))
    scaler = MinMaxScaler(feature_range=(weight_lb, 1))
    weights = scaler.fit_transform(weights)
    return attack_density, attack_lids_ratio, indices, normal_density, normal_lids_ratio, trunc_lid_ratio, weights
