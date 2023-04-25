import pandas as pd
from Functions.Plotting import plot_box

anal = 3
pred_model = "lasso"
n_repeats = 10
results_df = pd.read_csv("Results/miss-sim-{}/{}/results_nreps{}.csv".format(anal, pred_model, n_repeats), index_col=[0])
results_df['n_samples'] = results_df['n_samples'].astype(int)

# Plot results (one for each iv->dv cor:)
iv_dv_cor_list = [0, 0.1, 0.3, 0.5]

for cor in iv_dv_cor_list:
    print(cor)
    df = results_df[results_df['iv_dv_cor']==cor]
    plot_box(df=df, x='miss_perc', y='test_r2', group='n_samples',
             save_path="Results/miss-sim-{}/{}/Plots/".format(anal, pred_model), zero_line=True, fontsize=11,
             save_name="results_nreps{}_ivdv{}".format(n_repeats, cor),
             xlab="Missing Data Proportion", ylab="Prediction R squared (Test data)",
             title="Analysis {}, model= {}, K=30, n iter={}, DV R2={}".format(anal, pred_model, n_repeats, cor),
             leg_title="N samples")



print('done')