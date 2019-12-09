import argparse
import os
from sklearn.ensemble import IsolationForest as IF
import pandas as pd
from itertools import combinations as ncr
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description='Lookout algorithm')

    parser.add_argument("-b", "--budget", default=3, type=int,
                        help="Number of selected feature pairs. Default: 3.")

    parser.add_argument("-o", "--output_dir", default=os.path.join("out", "plots"), type=str,
                        help="Output directory for the plots. Default: out/plots/")

    return parser.parse_args()


def get_row_indices(S, feature_pairs):
    """Get row indices of given feature pairs"""
    return list(map(lambda elem: feature_pairs.index(elem), S))


def lookout(args):
    def f(S, outlier_scores):
        """Calculate total incrimination score of a given subset of plots (feature pairs)"""
        # Base case
        if not len(S):
            return 0

        # Only keep rows of plots being analyzed
        scores_S = outlier_scores[S, :]
        # Max score on each column (best plot for each outlier) and respective sum of all values
        return sum(np.max(scores_S, axis=0))

    # Number of plots to choose
    BUDGET = args.budget

    time = -timer()
    # Load dataset
    # Faster and easier alternative to test (worse results, of course)
    full_df = pd.read_csv("HTRU_2.csv", nrows=500)
    #full_df = pd.read_csv("HTRU_2.csv")

    # Isolate outliers and inliers
    # Points to later be drawn in BLACK
    inlier_df = full_df.loc[full_df['Class'] == 0]
    outlier_df = full_df.loc[full_df['Class'] ==
                             1].reset_index().drop(['index'], axis=1)

    # Remove target column, to not get mixed as a feature
    full_df.drop(columns=['Class'], inplace=True)
    inlier_df.drop(columns=['Class'], inplace=True)
    outlier_df.drop(columns=['Class'], inplace=True)

    # Get all available features and combine them 2 by 2
    all_features = list(full_df.columns)
    feature_pairs = list(ncr(all_features, 2))

    # Matrix with scores for all outliers on all feature-pair plots (row = plot, column = outlier)
    scores = None
    classifier = IF()  # Isolation Forest instance used to train and score outliers
    for feature_pair in feature_pairs:
        # Model for current feature pair
        classifier.fit(full_df[list(feature_pair)])
        scores = np.array([classifier.decision_function(outlier_df[list(feature_pair)]).tolist()]) if scores is None \
            else np.append(scores, [classifier.decision_function(outlier_df[list(feature_pair)]).tolist()], axis=0)

    # In Isolation Forest, negative scores are considered outliers and positive scores inliers. Original range is [-0.5, 0.5]
    # To ensure greedy approximation optimality we must ensure non negative range, i.e, convert scores to [0,1]
    # To do this, we flip the sign (so negatives become positives and outliers actually have better scores) and add 0.5
    transform_range = np.vectorize(lambda x: 0.5 - x)
    scores = transform_range(scores)

    # Plot selection using greedy heuristic approach (see paper for proof of near optimality)
    S = []  # Final plot selection
    while BUDGET > 0:
        # Only pairs that have not been selected already
        candidate_pairs = list(set(feature_pairs) - set(S))
        candidate_pairs_marginal_gains = []
        for candidate_pair in candidate_pairs:
            candidate_pairs_marginal_gains.append(
                f(get_row_indices(S+[candidate_pair], feature_pairs), scores) - f(get_row_indices(S, feature_pairs), scores))  # Marginal gain of current feature pair

        # Get max marginal gain, its index and retrieve respective feature pair
        S.append(candidate_pairs[candidate_pairs_marginal_gains.index(
            max(candidate_pairs_marginal_gains))])
        BUDGET = BUDGET - 1

    time = time + timer()

    print("Final selection: {}".format(S))
    print("Execution time: {0:.2f}s".format(time))

    # Actual Plotting
    # Tuple of (best_outliers, other_outliers) for each feature pair; IDS ONLY! MUST RETRIEVE FROM OUTLIER DATAFRAME
    outlier_points = []
    # For each selected plot, obtain list of outliers that are best explained by that feature pair (to be drawn in RED)
    # Remaining outliers to be drawn in BLUE
    for feature_pair in S:
        feature_pair_row_idx = get_row_indices(
            [feature_pair], feature_pairs)[0]
        outliers_max_plot_scores = np.max(scores, axis=0)
        feature_pair_plot_scores = scores[feature_pair_row_idx]
        # Returns boolean array checking if float values are close enough to be considered true #TODO Needs tuning?
        score_comparison = np.isclose(
            outliers_max_plot_scores, feature_pair_plot_scores)
        # IDs (in outliers dataframe) of outliers best explained by this feature pair
        best_outliers_ids = list(map(lambda x: x[0], filter(
            lambda y: y[1], enumerate(score_comparison.tolist()))))
        # shape property is a fast, safe way to extract number of rows in dataframe (x-shape)
        remaining_outliers_ids = list(
            set(range(outlier_df.shape[0])) - set(best_outliers_ids))
        outlier_points.append((best_outliers_ids, remaining_outliers_ids))

    # Plotting the chosen features
    for feature_pair, outliers_p in zip(S, outlier_points):
        # Adding inliers
        plot_df = inlier_df.copy()
        plot_df = plot_df[list(feature_pair)]
        plot_df['class'] = 'inlier'
        plot_df['point_size'] = 30

        # Other outliers
        other_outliers = outlier_df.iloc[outliers_p[1]]
        other_outliers = other_outliers[list(feature_pair)]
        other_outliers['class'] = 'other'
        other_outliers['point_size'] = 45

        # Explained outliers
        best_outliers = outlier_df.iloc[outliers_p[0]]
        best_outliers = best_outliers[list(feature_pair)]
        best_outliers['class'] = 'best'
        best_outliers['point_size'] = 45

        # Joining all the dataframes
        plot_df = plot_df.append(best_outliers)
        plot_df = plot_df.append(other_outliers)

        # Actual Plotting
        f, ax = plt.subplots(figsize=(6.5, 6.5))
        sns.scatterplot(x=feature_pair[0], y=feature_pair[1],
                        hue="class", size="point_size",
                        palette=sns.color_palette(
                            ["#000000", "#FF0000", "#0000FF"]),
                        linewidth=1, legend='full', alpha=0.7,
                        edgecolor='black',
                        data=plot_df, ax=ax)

        # Saving plots
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        plt.savefig('%s/%s_%s.png' % (args.output_dir, *feature_pair))


if __name__ == '__main__':
    lookout(parse_args())
