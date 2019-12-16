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

    parser.add_argument("-d", "--dataset", default=0, type=int,
                        help="Dataset to run algorithm. Values 0 (HTRU2), else (CTG). Default 0")

    parser.add_argument("-o", "--output_dir", default=os.path.join("out", "plots"), type=str,
                        help="Output directory for the plots. Default: out/plots/")
    
    parser.add_argument("-n", "--new", action="store_true",
                        help="Run modified algorithm (with alfa weighting)")
    
    parser.add_argument("-a", "--alfa", default=0.95, type=float,
                        help="Alfa value to weigh both gain components. Default 0.9")

    return parser.parse_args()


def get_row_indices(S, feature_pairs):
    """Get row indices of given feature pairs"""
    return list(map(lambda elem: feature_pairs.index(elem), S))

def get_num_features(S):
    """Counts number of unique features in given pair combination subset"""
    if not len(S):
        return 1
        
    return len(np.unique(np.array(list(sum(S, ())))))

def f(S, outlier_scores):
        """Calculate total incrimination score of a given subset of plots (feature pairs)"""
        # Base case
        if not len(S):
            return 0

        # Only keep rows of plots being analyzed
        scores_S = outlier_scores[S, :]
        # Max score on each column (best plot for each outlier) and respective sum of all values
        return sum(np.max(scores_S, axis=0))

def get_marginal_gain(S, candidate_pair, feature_pairs, scores, args):
    """Calculates marginal gain of a given feature pair in relation to current selected plots"""
    if args.new:
        return args.alfa * (f(get_row_indices(S+[candidate_pair], feature_pairs), scores) - f(get_row_indices(S, feature_pairs), scores)) + \
                (1-args.alfa) * (get_num_features(S+[candidate_pair]) / get_num_features(S))
    else:
        return f(get_row_indices(S+[candidate_pair], feature_pairs), scores) - f(get_row_indices(S, feature_pairs), scores)

def get_palette(df):
    classes = df['class'].unique()
    if len(classes) == 3:
        return sns.color_palette(["#000000", "#FF0000", "#0000FF"])
    elif 'other' in classes: #Only 2 classes, inlier and other outliers
        return sns.color_palette(["#000000", "#0000FF"])
    else: #Only 2 classes, inlier and best outliers
        return sns.color_palette(["#000000", "#FF0000"])

def lookout(args):
    # Number of plots to choose
    BUDGET = args.budget

    time = -timer()
    # Load dataset
    # Faster and easier alternative to test (worse results, of course)
    #full_df = pd.read_csv("HTRU_2.csv", nrows=500)
    #full_df = pd.read_csv("HTRU_2.csv")
    if args.dataset == 0:
        full_df = pd.read_csv("HTRU_2_filtered.csv") #outlier proportion: ~0.002
    else:
        full_df = pd.read_csv("CTG_Filtered.csv") #outlier proportion: ~0.1

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
    # Isolation Forest instance used to train and score outliers
    if args.dataset == 0:
        classifier = IF(max_samples=64, contamination=0.02) #for HTRU dataset
    else:
        classifier = IF(max_samples=64, contamination=0.1) #for CTG dataset
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
            # Marginal gain of current feature pair
            candidate_pairs_marginal_gains.append(get_marginal_gain(S, candidate_pair, feature_pairs, scores, args))

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
        # Returns boolean array checking if float values are close enough to be considered true
        score_comparison = np.isclose(
            feature_pair_plot_scores, outliers_max_plot_scores, atol=1e-6)
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
        plot_df['point_size'] = 25

        # Other outliers
        other_outliers = outlier_df.iloc[outliers_p[1]]
        other_outliers = other_outliers[list(feature_pair)]
        other_outliers['class'] = 'other'
        other_outliers['point_size'] = 25

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
                        palette=get_palette(plot_df),
                        linewidth=1, legend='full', alpha=0.7,
                        edgecolor='black',
                        data=plot_df, ax=ax)
        plt.autoscale(True)

        # Saving plots
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        plt.savefig('%s/%s_%s.png' % (args.output_dir, *feature_pair))


if __name__ == '__main__':
    lookout(parse_args())
