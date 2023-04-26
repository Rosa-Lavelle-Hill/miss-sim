import pandas as pd
from Functions.Plotting import plot_box

anal = 3
pred_model = "lasso"
n_repeats = 10
results_df = pd.read_csv("Results/miss-sim-{}/{}/results_nreps{}.csv".format(anal, pred_model, n_repeats), index_col=[0])
results_df['n_samples'] = results_df['n_samples'].astype(int)

# Plot results (one for each iv->dv cor:)
r2_list = [0, 0.1, 0.3, 0.5]

for r2_param in r2_list:
    print(r2_param)
    df = results_df[results_df['r2_param'] == r2_param]
    plot_box(df=df, x='miss_perc', y='test_r2', group='n_samples',
             save_path="Results/miss-sim-{}/{}/Plots/".format(anal, pred_model), fontsize=11,
             save_name="results_nreps{}_r2{}".format(n_repeats, r2_param),
             xlab="Missing Data Proportion", ylab="Prediction R squared (Test data)",
             title="Analysis {}, model= {}, K=30, n iter={}, R2={}".format(anal, pred_model, n_repeats, r2_param),
             leg_title="N samples",
             r2=r2_param)



print('done')