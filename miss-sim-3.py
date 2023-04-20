# miss-sim 3: Only X has missing values, only X used to impute X

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import Lasso
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from Functions.Plotting import plot_box
from Params.Grids import enet_param_grid, lasso_param_grid, rf_param_grid

anal = 3

# choose prediction model (enet, lasso, rf):
pred_model = "enet"

# Define the parameters
n_repeats = 100
K = 30
iv_cor = 0.3
mean = 0
sd = 1
test_size = 0.5
max_iter_imp = 10
cv = 5
scoring = "r2"
decimal_places = 2
fixed_seed = 93

# Params to loop through:
K_list = [K]
n_samples_list = [100, 300, 1000]
miss_perc_list = [0.1, 0.3, 0.5]
# ^this is miss percentage for each variable

results_dict = {}
iter=1
for K in K_list:
    for n_samples in n_samples_list:
        for miss_perc in miss_perc_list:
            for seed in range(0, n_repeats):

                np.random.seed(seed)

                nomiss_perc = 1-miss_perc

                # ----------- Simulate and Impute -----------

                # Define the mean and standard deviation for each variable
                means = [mean] * K
                stds = [sd] * K

                # Define the correlation matrix
                ones = np.ones((K, K))
                corr_matrix = iv_cor * ones + (1 - iv_cor) * np.eye(K)

                # Generate the simulated X data with _% missing
                X = np.random.multivariate_normal(mean=means, cov=corr_matrix, size=n_samples)
                missing_mask = np.random.choice([True, False], size=X.shape, p=[miss_perc, nomiss_perc])
                X_missing = np.where(missing_mask, np.nan, X)

                # Separately generate y with _% missing
                y = np.random.normal(loc=mean, scale=sd, size=n_samples)

                # Join X and y
                X_col_names = []
                for i in list(range(1, X.shape[1]+1)):
                    n = "X{}".format(i)
                    X_col_names.append(n)

                X_missing = pd.DataFrame(X_missing, columns=X_col_names)
                y = pd.DataFrame(y, columns=['y'])

                # Check correlations (ignoring NAs)
                if iter == 1:
                    X_and_y_miss = pd.concat([X_missing, y], axis=1)
                    save_path = "Outputs/miss-sim-{}/Descriptives/".format(anal)
                    data_cor_matrix = pd.DataFrame(X_and_y_miss.corr())
                    data_cor_matrix.to_csv(save_path + "X_and_y_cor_ignoring_NAs_nreps{}.csv".format(n_repeats))

                # Define imputer (default = BayesianRidge())
                imp_mean = IterativeImputer(random_state=seed, max_iter=max_iter_imp, imputation_order='ascending')

                # Impute X only
                X_imp = imp_mean.fit_transform(X_missing)
                X_imp = pd.DataFrame(X_imp, columns=X_missing.columns)

                # Join imputed X to y
                X_and_y_imp = pd.concat([X_imp, y], axis=1)

                # ----------- Train Model -----------

                # Redefine X and y from X_and_y
                y = X_and_y_imp['y']
                X = X_and_y_imp.drop('y', axis=1)

                # Split train and test (same random seed so constant stable comparison)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

                # Define model
                if pred_model == "enet":
                    model = ElasticNet(random_state=fixed_seed)
                    param_grid = enet_param_grid
                elif pred_model == "lasso":
                    model = Lasso(alpha=1, max_iter=500, random_state=fixed_seed)
                    param_grid = lasso_param_grid
                elif pred_model == "rf":
                    model = RandomForestRegressor(random_state=fixed_seed)
                    param_grid = rf_param_grid
                else:
                    print("Error: please define pred_model param as one of 'enet', 'lasso', or 'rf'")
                    breakpoint()

                # CV on train data to tune hyper-parameters
                grid_search = GridSearchCV(estimator=model,
                                                       param_grid=param_grid,
                                                       cv=cv,
                                                       scoring=scoring,
                                                       refit=True,
                                                       verbose=2,
                                                       n_jobs=2)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                cv_r2 = grid_search.best_score_

                # ----------- Predict Test Data -----------

                # Predict y test
                y_pred = model.predict(X_test)
                test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
                print(test_r2)

                dict = {"miss_perc": miss_perc,
                          "K": K,
                          "n_samples": n_samples,
                          "cv_r2": cv_r2,
                          "test_r2": test_r2,
                          "seed": seed}

                results_dict[iter] = dict
                iter = iter +1

results_df = pd.DataFrame.from_dict(results_dict).T
results_df.to_csv("Results/miss-sim-{}/{}/results_nreps{}.csv".format(anal, pred_model, n_repeats))

# Plot results

plot_box(df=results_df, x='miss_perc', y='test_r2', group='n_samples',
         save_path="Results/miss-sim-{}/{}/Plots/".format(anal, pred_model), zero_line=True, fontsize=11,
         save_name="results_nreps{}".format(n_repeats),
         xlab="Missing Data Proportion", ylab="Prediction R squared (Test data)",
         title="Analysis {}, model= {}, K=30, n iter={}".format(anal, pred_model, n_repeats), leg_title="N samples")

print('done')