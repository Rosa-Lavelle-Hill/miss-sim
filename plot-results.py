import pandas as pd
from Functions.Plotting import plot_box

n_repeats = 100
results_df = pd.read_csv("Results/results_nreps{}.csv".format(n_repeats), index_col=[0])
results_df['n_samples'] = results_df['n_samples'].astype(int)

plot_box(df=results_df, x='miss_perc', y='test_r2', group='n_samples',
         save_path="Results/Plots/", zero_line=True, fontsize=11,
         save_name="results_nreps{}".format(n_repeats),
         xlab="Missing Data Proportion", ylab="Prediction R squared (Test data)",
         title="K=30", leg_title="N samples")

print('done')