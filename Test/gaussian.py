import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import itertools

def gaussian_tuning(X, max_range):
    lowest_bic = np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        print('\n'+cv_type)
        i = 0
        for n_components in range(1, max_range + 1):
            print(i/max_range)
            i += 1
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm, bic


def plot_bic_scores(bic, max_range):
    # Plot the BIC scores
    cv_types = ['spherical','tied','diag','full']
    color_iter = itertools.cycle(['navy','turquoise','cornflowerblue','darkorange'])
    bars = []
    n_components_range = range(1, max_range + 1)
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                    (i + 1) * len(n_components_range)], width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)