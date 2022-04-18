import numpy as np
import matplotlib.pyplot as plt


def plot_multiclass_reliability_diagram(
    y_true, p_pred, n_bins=15, title=None, fig=None, ax=None, legend=True
):
    """
    y_true:  needs to be (n_samples,)
    """
    if fig is None and ax is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    y_true = y_true.flatten()
    p_pred = p_pred.flatten()

    bin_size = 1 / n_bins
    centers = np.linspace(bin_size / 2, 1.0 - bin_size / 2, n_bins)
    true_proportion = np.zeros(n_bins)

    pred_mean = np.zeros(n_bins)
    for i, center in enumerate(centers):
        if i == 0:
            # First bin include lower bound
            bin_indices = np.where(
                np.logical_and(
                    p_pred >= center - bin_size / 2, p_pred <= center + bin_size / 2
                )
            )
        else:
            bin_indices = np.where(
                np.logical_and(
                    p_pred > center - bin_size / 2, p_pred <= center + bin_size / 2
                )
            )
        true_proportion[i] = np.mean(y_true[bin_indices])
        pred_mean[i] = np.mean(p_pred[bin_indices])

    ax.bar(
        centers,
        true_proportion,
        width=bin_size,
        edgecolor="black",
        color="blue",
        label="True class prop.",
    )
    ax.bar(
        centers,
        true_proportion - pred_mean,
        bottom=pred_mean,
        width=bin_size / 2,
        edgecolor="red",
        color="#ffc8c6",
        alpha=1,
        label="Gap pred. mean",
    )
    if legend:
        ax.legend()

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
