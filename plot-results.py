import pandas as pd
from Functions.Plotting import plot_box

anal = 2
pred_model = "enet"
n_repeats = 10
results_df = pd.read_csv("Results/miss-sim-{}/{}/results_nreps{}.csv".format(anal, pred_model, n_repeats), index_col=[0])
results_df['n_samples'] = results_df['n_samples'].astype(int)

plot_box(df=results_df, x='miss_perc', y='test_r2', group='n_samples',
         save_path="Results/miss-sim-{}/{}/Plots/".format(anal, pred_model), zero_line=True, fontsize=11,
         save_name="results_nreps{}".format(n_repeats),
         xlab="Missing Data Proportion", ylab="Prediction R squared (Test data)",
         title="Analysis {}, model= {}, K=30, n iter={}".format(anal, pred_model, n_repeats), leg_title="N samples")

print('done')