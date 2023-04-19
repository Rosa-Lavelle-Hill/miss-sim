import matplotlib.pyplot as plt
import seaborn as sns

def plot_box(df, x, y, group, save_path, save_name,
                  xlab, ylab, title, y_lim=None, legend_under=False,
                  zero_line=False, leg_title="",
                  fontsize=12, palette="Paired", figsize=(12, 12)):

    sns.set_palette(palette)
    plt.figure(figsize=figsize)
    sns.catplot(x=x, y=y, hue=group, kind="box", data=df, legend=False)
    if y_lim:
        plt.ylim(y_lim)

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)

    if legend_under == True:
        legend_fontsize = fontsize - 4
        plt.legend(loc='lower left',
                   bbox_to_anchor=(-0.15, -0.5),
                   fontsize=legend_fontsize,
                   title=leg_title)
    else:
        plt.legend(title=leg_title)

    if zero_line == True:
        plt.axhline(y=0, color='grey', linestyle='--')

    plt.tight_layout()

    plt.savefig(save_path + save_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()