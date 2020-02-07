import time

import numpy as np
from sklearn.linear_model import RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.preprocessing import MinMaxScaler

from utils.utilities import get_lids_with_neighbors, get_model, clean_lid_ratios


def trimclf(x, y, count, alpha, learner):
    newinds = []
    it = 0
    toterr = 10000
    lasterr = 20000
    length = x.shape[0]

    # start the timer
    t = time.process_time()
    inds = sorted(np.random.permutation(length))[:count]
    clf = learner(alpha=alpha, fit_intercept=True)

    while sorted(inds) != sorted(newinds) and it < 400 and lasterr - toterr > 1e-5:
        newinds = inds[:]
        lasterr = toterr
        subx = x[inds]
        suby = y[inds]
        clf.fit(subx, suby)
        preds = clf.predict(x) - y
        residvec = np.square(preds)

        residtopns = sorted([(residvec[i], i) for i in range(length)])[:count]
        resid = [val[1] for val in residtopns]
        topnresid = [val[0] for val in residtopns]

        # set inds to indices of n largest values in error
        inds = sorted(resid)
        # recompute error
        toterr = sum(topnresid)
        it += 1

    # end the timer
    elapsed_time = time.process_time() - t
    return clf, elapsed_time


def nlid_lr(x, y, normal_sample_size, alpha, learner, poison_perc, cutoff, k):
    # This defense cannot be run when there is no attack
    # if no attack
    if normal_sample_size == x.shape[0]:
        t = time.process_time()
        clf = learner(alpha=alpha, fit_intercept=True)
        clf.fit(x, y)
        elapsed_time = time.process_time() - t
        return clf, elapsed_time

    weight_lb = 0
    # if there is an attack
    all_data = np.hstack((y.reshape(-1, 1), x))

    # start the timer
    t = time.process_time()

    all_lids, lid_ratio = get_lids_with_neighbors(all_data, k)
    attack_density, attack_lids_ratio, indices, normal_density, normal_lids_ratio, trunc_lid_ratio, weights = clean_lid_ratios(
        cutoff, lid_ratio, normal_sample_size, weight_lb)

    clf = learner(alpha=alpha, fit_intercept=True)
    clf.fit(x[indices, :], y[indices], sample_weight=weights.ravel())

    # end the timer
    elapsed_time = time.process_time() - t

    return clf, elapsed_time


def nlid_linear(x, y, alpha, learner, cutoff, k):
    all_data = np.hstack((y.reshape(-1, 1), x))

    # start the timer
    t = time.process_time()

    all_lids, lid_ratio = get_lids_with_neighbors(all_data, k)
    indices = np.argwhere((lid_ratio < cutoff) & (lid_ratio != np.inf)).ravel()
    lid_ratio = lid_ratio[indices]

    # scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    weights = 1 - scaler.fit_transform(lid_ratio.reshape(-1, 1))
    # set any negative values to 0
    weights[np.where(weights < 0)] = 0

    clf = learner(alpha=alpha, fit_intercept=True)
    clf.fit(x[indices, :], y[indices], sample_weight=weights.ravel())

    # end the timer
    elapsed_time = time.process_time() - t

    return clf, elapsed_time


def nlid_nonlinear(x, y, alpha, learner, cutoff, k, func_type='convex'):
    all_data = np.hstack((y.reshape(-1, 1), x))

    # start the timer
    t = time.process_time()

    all_lids, lid_ratio = get_lids_with_neighbors(all_data, k)
    indices = np.argwhere((lid_ratio < cutoff) & (lid_ratio != np.inf)).ravel()
    lid_ratio = lid_ratio[indices]

    # scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    weights = scaler.fit_transform(lid_ratio.reshape(-1, 1))

    if func_type == 'concave':
        f = lambda p: 1 - p ** 2
        transformed_weights = f(weights)
    else:
        f = lambda p: 1 - ((2 * p - p ** 2) ** 0.07)
        transformed_weights = f(weights)

    # set negative values to 0
    transformed_weights[np.where(transformed_weights < 0)] = 0
    clf = learner(alpha=alpha, fit_intercept=True)
    clf.fit(x[indices, :], y[indices], sample_weight=transformed_weights.ravel())

    # end the timer
    elapsed_time = time.process_time() - t
    return clf, elapsed_time


def nn_regression(x, y, dataset):
    # start the timer
    t = time.process_time()
    # Each dataset has a different model
    model, epochs, batch_size = get_model(dataset, x.shape[1])
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # end the timer
    elapsed_time = time.process_time() - t
    return model, elapsed_time


def nn_regression_lid(x, y, dataset, normal_sample_size, cutoff, k):
    # This defense cannot be run when there is no attack
    # if no attack
    if normal_sample_size == x.shape[0]:
        return nn_regression(x, y, dataset)

    weight_lb = 0
    # if there is an attack
    all_data = np.hstack((y.reshape(-1, 1), x))

    # start the timer
    t = time.process_time()

    all_lids, lid_ratio = get_lids_with_neighbors(all_data, k)
    attack_density, attack_lids_ratio, indices, normal_density, normal_lids_ratio, trunc_lid_ratio, weights = clean_lid_ratios(
        cutoff, lid_ratio, normal_sample_size, weight_lb)

    # Each dataset has a different model
    model, epochs, batch_size = get_model(dataset, x.shape[1])
    model.fit(x[indices, :], y[indices], sample_weight=weights.ravel(), epochs=epochs, batch_size=batch_size, verbose=0)

    # end the timer
    elapsed_time = time.process_time() - t
    return model, elapsed_time


def nn_regression_lid_nonlinear(x, y, dataset, cutoff, k, func_type='convex'):
    all_data = np.hstack((y.reshape(-1, 1), x))

    # start the timer
    t = time.process_time()

    all_lids, lid_ratio = get_lids_with_neighbors(all_data, k)
    indices = np.argwhere((lid_ratio < cutoff) & (lid_ratio != np.inf)).ravel()
    lid_ratio = lid_ratio[indices]

    # scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    weights = scaler.fit_transform(lid_ratio.reshape(-1, 1))

    if func_type == 'concave':
        f = lambda p: 1 - p ** 3
        transformed_weights = f(weights)
    else:
        f = lambda p: 1 - ((2 * p - p ** 2) ** 0.07)
        transformed_weights = f(weights)

    transformed_weights[np.where(transformed_weights < 0)] = 0
    # Each dataset has a different model
    model, epochs, batch_size = get_model(dataset, x.shape[1])
    model.fit(x[indices, :], y[indices], sample_weight=transformed_weights.ravel(), epochs=epochs,
              batch_size=batch_size, verbose=0)

    # end the timer
    elapsed_time = time.process_time() - t
    return model, elapsed_time


def huber_model(x, y, epsilon_values, alpha):
    scores = []
    best_clf, best_count, best_time, best_score, = None, None, None, 0

    for eps in epsilon_values:
        t = time.process_time()
        clf = HuberRegressor(epsilon=eps, max_iter=1000, alpha=alpha)
        clf.fit(x, y)
        elapsed_time = time.process_time() - t
        score = clf.score(x[~clf.outliers_], y[~clf.outliers_])
        scores.append(score)
        if score > best_score:
            best_clf, best_eps, best_score, best_time = clf, eps, score, elapsed_time
    return best_clf, best_time


def ransac_model(x, y, model, count_values):
    scores = []
    best_clf, best_count, best_time, best_score, = None, None, None, 0

    for count in count_values:
        t = time.process_time()
        reg_ransac = RANSACRegressor(model, min_samples=count)
        reg_ransac.fit(x, y)
        elapsed_time = time.process_time() - t
        score = reg_ransac.score(x[reg_ransac.inlier_mask_], y[reg_ransac.inlier_mask_])
        scores.append(score)
        if score > best_score:
            best_clf, best_count, best_score, best_time = reg_ransac, count, score, elapsed_time
    return best_clf, best_time


def theilsen_model(x, y, n_subsamples):
    best_clf, best_breakdown = None, 0

    t = time.process_time()
    for samples_size in n_subsamples:
        reg_ts = TheilSenRegressor(n_subsamples=samples_size, copy_X=True)
        reg_ts.fit(x, y)
        breakdown = reg_ts.breakdown_
        if breakdown > best_breakdown:
            best_clf, best_breakdown = reg_ts, breakdown

    elapsed_time = time.process_time() - t
    return best_clf, elapsed_time
