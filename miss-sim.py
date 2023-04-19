import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNet
from Params.Grids import enet_param_grid

# Set the random seed for reproducibility
seed = 93
np.random.seed(seed)

# Define the parameters
n_samples = 1000
K = 10
mean = 0
sd = 1
miss_perc = 0.1
nomiss_perc = 1-miss_perc
iv_cor = 0.3
test_size = 0.2
max_iter_imp = 10
cv = 5
scoring = "r2"
decimal_places = 2

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
missing_mask = np.random.choice([True, False], size=y.shape, p=[miss_perc, nomiss_perc])
y_missing = np.where(missing_mask, np.nan, y)

# Join X and y
X_col_names = []
for i in list(range(1, X.shape[1]+1)):
    n = "X{}".format(i)
    X_col_names.append(n)

X_missing = pd.DataFrame(X_missing, columns=X_col_names)
y_missing = pd.DataFrame(y_missing, columns=['y'])
X_and_y_miss = pd.concat([X_missing, y_missing], axis=1)

# Check correlations (ignoring NAs)
save_path = "Outputs/Descriptives/"
data_cor_matrix = pd.DataFrame(X_and_y_miss.corr())
data_cor_matrix.to_csv(save_path + "X_and_y_cor_ignoring_NAs.csv")

# Define imputer (default = BayesianRidge())
imp_mean = IterativeImputer(random_state=seed, max_iter=max_iter_imp, imputation_order='ascending')

# Impute X and y together
X_and_y_imp = imp_mean.fit_transform(X_and_y_miss)
X_and_y_imp = pd.DataFrame(X_and_y_imp, columns=X_and_y_miss.columns)

# ----------- Train Elastic Net -----------

# Redefine X and y from X_and_y
y = X_and_y_imp['y']
X = X_and_y_imp.drop('y', axis=1)

# Split train and test (same random seed so constant stable comparison)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

# Define models
enet = ElasticNet()

# CV on train data to tune hyper-parameters
grid_search = GridSearchCV(estimator=enet,
                                       param_grid=enet_param_grid,
                                       cv=cv,
                                       scoring=scoring,
                                       refit=True,
                                       verbose=1,
                                       n_jobs=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
enet.set_params(**best_params)
enet.fit(X_train, y_train)

# Predict y test
y_pred = enet.predict(X_test)
test_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
print(test_r2)


print('done')