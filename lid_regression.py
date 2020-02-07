import random
import warnings

import pandas as pd
from sklearn.linear_model import Ridge

from defenses.defenses import *
from utils.utilities import classification_perf, unison_shuffled_copies

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=FutureWarning)

random.seed(0)
np.random.seed(0)

ATTK_NAME = 'ridge_randflip'  # use 'ridge_randflipnobd' for IFLip
DATASET = 'house'
ALPHA = 0.02726798053  # regularization parameter for Ridge

LEARNER = Ridge

lid_ratio_cutoffs = {'house': {'ridge_randflip': 5, 'mlsg': 5, 'ridge_randflipnobd': 5},
                     'loan': {'ridge_randflip': 4, 'ridge_randflipnobd': 4, 'mlsg': 4},
                     'grid': {'ridge_randflip': 5, 'ridge_randflipnobd': 4},
                     'machine': {'ridge_randflip': 5, 'ridge_randflipnobd': 4},
                     'omnet': {'ridge_randflip': 2.5, 'ridge_randflipnobd': 2.5}}

# lid parameters
k = 35  # k value for N-LID calculation

# Dataframe to hold results
column_names = ['scenario', 'cv_set', 'attack_index', 'MSE', 'time']
df = pd.DataFrame(columns=column_names)

poison_percentages = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.20])

cv_range_start = 1
cv_range_end = 6

# Choose the defenses you would like to execute
# no_defense - ridge without a defense
# nn_no_defense - NNR without a defense
# nn_lid_lr - NNR with N-LID LR defense
# nn_lid_convex - NNR with N-LID CVX defense
# nlid_lr - ridge with N-LID LR defense
# nlid_convex - ridge with N-LID CVX defense
defenses = ['no_defense', 'nn_no_defense', 'nn_lid_lr', 'nn_lid_convex', 'nlid_lr', 'trim',
            'nlid_convex', 'RANSACRegressor', 'HuberRegressor']

defenses = ['no_defense', 'nlid_lr', 'trim',
            'nlid_convex']
for cv in range(cv_range_start, cv_range_end):

    # load the training data
    train_x = np.load('./datasets/attacked/{}/{}/CV_{}/trainx.npy'.format(DATASET, ATTK_NAME, cv))
    train_y = np.load('./datasets/attacked/{}/{}/CV_{}/trainy.npy'.format(DATASET, ATTK_NAME, cv))

    normal_sample_size = train_x.shape[0]

    # load the test data
    test_x = np.load('./datasets/attacked/{}/{}/CV_{}/testx.npy'.format(DATASET, ATTK_NAME, cv))
    test_y = np.load('./datasets/attacked/{}/{}/CV_{}/testy.npy'.format(DATASET, ATTK_NAME, cv))

    ###############################################################
    # load attack data
    ###############################################################
    for attack_indx in range(0, 6):
        poison_perc = poison_percentages[attack_indx]

        if attack_indx != 0:
            pois_res = np.load(
                './datasets/attacked/{}/{}/CV_{}/poisres_{:.2f}.npy'.format(DATASET, ATTK_NAME, cv, poison_perc))
            pois_res_y = np.load(
                './datasets/attacked/{}/{}/CV_{}/poisresy_{:.2f}.npy'.format(DATASET, ATTK_NAME, cv, poison_perc))

            X = np.vstack((train_x, pois_res))
            y = np.vstack((train_y.reshape(-1, 1), pois_res_y.reshape(-1, 1)))
        else:
            X = train_x
            y = train_y

        ###############################################################
        # no defense
        ###############################################################
        if 'no_defense' in defenses:
            t = time.process_time()

            reg_learner = LEARNER(alpha=ALPHA, fit_intercept=True)
            reg_learner.fit(np.copy(X), np.copy(y))

            elapsed_time = time.process_time() - t

            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'no_defense', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # RANSAC
        ###############################################################
        if 'RANSACRegressor' in defenses:
            X_copy, y_copy = unison_shuffled_copies(np.copy(X), np.copy(y))
            count_values = [0.1, 0.2, 0.3, 0.4]
            ridge_model = LEARNER(alpha=ALPHA, fit_intercept=True)
            reg_learner, elapsed_time = ransac_model(X_copy, y_copy, ridge_model, count_values)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'RANSACRegressor', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # Huber
        ###############################################################
        if 'HuberRegressor' in defenses:
            X_copy, y_copy = unison_shuffled_copies(np.copy(X), np.copy(y))
            epsilon_values = [1.35, 1.5, 1.75, 1.9]
            reg_learner, elapsed_time = huber_model(X_copy, y_copy, epsilon_values, ALPHA)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'HuberRegressor', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # TheilSenRegressor
        ###############################################################
        if 'TheilSenRegressor' in defenses:
            X_copy, y_copy = unison_shuffled_copies(np.copy(X), np.copy(y))
            n_subsamples = [50, 100, 200, 300]
            reg_learner, elapsed_time = theilsen_model(X_copy, y_copy, n_subsamples)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'TheilSenRegressor', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # N-LID LR
        ###############################################################
        if 'nlid_lr' in defenses:
            reg_learner, elapsed_time = nlid_lr(np.copy(X), np.copy(y), normal_sample_size, ALPHA, LEARNER,
                                                poison_perc, lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME), k)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nlid_lr', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # N-LID linear
        ###############################################################
        if 'nlid_linear' in defenses:
            reg_learner, elapsed_time = nlid_linear(np.copy(X), np.copy(y), ALPHA, LEARNER,
                                                    lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME), k)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nlid_linear', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # N-LID convex
        ###############################################################
        if 'nlid_convex' in defenses:
            reg_learner, elapsed_time = nlid_nonlinear(np.copy(X), np.copy(y), ALPHA, LEARNER,
                                                       lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME),
                                                       k, 'convex')
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nlid_convex', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # N-LID concave
        ###############################################################
        if 'lid_ratio_concave' in defenses:
            reg_learner, elapsed_time = nlid_nonlinear(np.copy(X), np.copy(y), ALPHA, LEARNER,
                                                       lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME),
                                                       k, 'concave')
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'lid_ratio_concave', cv, attack_indx, elapsed_time, column_names),
                ignore_index=True)

        ###############################################################
        # NN no defense
        ###############################################################
        if 'nn_no_defense' in defenses:
            reg_learner, elapsed_time = nn_regression(np.copy(X), np.copy(y), DATASET)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nn_no_defense', cv, attack_indx, elapsed_time,
                                    column_names),
                ignore_index=True)

        ###############################################################
        # NN N-LID LR
        ###############################################################
        if 'nn_lid_lr' in defenses:
            reg_learner, elapsed_time = nn_regression_lid(np.copy(X), np.copy(y), DATASET, normal_sample_size,
                                                          lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME), k)
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nn_lid_lr', cv, attack_indx, elapsed_time,
                                    column_names),
                ignore_index=True)

        ###############################################################
        # NN N-LID CVX
        ###############################################################
        if 'nn_lid_convex' in defenses:
            reg_learner, elapsed_time = nn_regression_lid_nonlinear(np.copy(X), np.copy(y), DATASET,
                                                                    lid_ratio_cutoffs.get(DATASET).get(ATTK_NAME),
                                                                    k, 'convex')
            y_hat = reg_learner.predict(test_x)
            df = df.append(
                classification_perf(test_y, y_hat, 'nn_lid_convex', cv, attack_indx, elapsed_time,
                                    column_names),
                ignore_index=True)

        ###############################################################
        # trim defense
        ###############################################################
        if 'trim' in defenses:
            X_copy, y_copy = unison_shuffled_copies(np.copy(X), np.copy(y))

            reg_learner, elapsed_time = trimclf(X_copy, y_copy, normal_sample_size, ALPHA, LEARNER)
            y_hat = reg_learner.predict(test_x)
            df = df.append(classification_perf(test_y, y_hat, 'trim', cv, attack_indx, elapsed_time, column_names),
                           ignore_index=True)

# Dataframe to hold results
column_names = ['scenario', 'attack_index', 'avg_MSE', 'std_MSE', 'avg_time', 'std_time']
df_results = pd.DataFrame(columns=column_names)

for defense in defenses:
    learner_df = df[df.scenario == defense]
    for attack_indx in range(0, 6):
        temp_df = learner_df[learner_df.attack_index == attack_indx]
        mean_mse = temp_df['MSE'].mean()
        std_mse = temp_df['MSE'].std()
        mean_time = temp_df['time'].mean()
        std_time = temp_df['time'].std()
        df_results = df_results.append(
            pd.DataFrame([[defense, attack_indx, mean_mse, std_mse, mean_time, std_time]], columns=column_names))

full_results = 'res_{}_{}.txt'.format(DATASET, ATTK_NAME)
summary_results = 'sum_{}_{}.txt'.format(DATASET, ATTK_NAME)

with open(full_results, 'w+') as f:
    df.to_csv(f, encoding='utf-8', index=False)

with open(summary_results, 'w+') as f:
    df_results.to_csv(f, encoding='utf-8', index=False)
