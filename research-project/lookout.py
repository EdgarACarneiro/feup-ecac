from sklearn.ensemble import IsolationForest as IF
import pandas as pd
from itertools import combinations as ncr
import numpy as np
from timeit import default_timer as timer

#Get row indices of given feature pairs
def get_row_indices(S, feature_pairs):
    return list(map(lambda elem: feature_pairs.index(elem), S))

#Calculate total incrimination score of a given subset of plots (feature pairs) s
def f(S, outlier_scores):
    #Base case
    if not len(S):
        return 0

    scores_S = outlier_scores[S, :] #Only keep rows of plots being analyzed
    return sum(np.max(scores_S,axis=0)) #Max score on each column (best plot for each outlier) and respective sum of all values

#Number of plots to choose
BUDGET = 3

time = -timer()
#Load dataset
full_df = pd.read_csv("HTRU_2.csv", nrows=500) #Faster and easier alternative to test (worse results, of course)
#full_df = pd.read_csv("HTRU_2.csv")

#Isolate outliers and inliers
inlier_df = full_df.loc[full_df['Class'] == 0] #Points to later be drawn in BLACK
outlier_df = full_df.loc[full_df['Class'] == 1]

#Remove target column, to not get mixed as a feature
full_df.drop(columns=['Class'], inplace=True)
inlier_df.drop(columns=['Class'], inplace=True)
outlier_df.drop(columns=['Class'], inplace=True)

#Get all available features and combine them 2 by 2
all_features = list(full_df.columns)
feature_pairs = list(ncr(all_features, 2))

#Matrix with scores for all outliers on all feature-pair plots (row = plot, column = outlier)
scores = None
classifier = IF() #Isolation Forest instance used to train and score outliers
for feature_pair in feature_pairs:
    classifier.fit(full_df[list(feature_pair)]) #Model for current feature pair
    scores = np.array([classifier.decision_function(outlier_df[list(feature_pair)]).tolist()]) if scores is None \
                else np.append(scores, [classifier.decision_function(outlier_df[list(feature_pair)]).tolist()], axis=0)

#In Isolation Forest, negative scores are considered outliers and positive scores inliers. 
#We flip the sign so more negative values are actually more relevant, i.e., mean that that feature pair better helps explain given outlier
scores = np.negative(scores)

#Plot selection using greedy heuristic approach (see paper for proof of near optimality)
S = [] #Final plot selection
while BUDGET > 0:
    candidate_pairs = list(set(feature_pairs) - set(S)) #Only pairs that have not been selected already
    candidate_pairs_marginal_gains = []
    for candidate_pair in candidate_pairs:
        candidate_pairs_marginal_gains.append(\
                        f(get_row_indices(S+[candidate_pair], feature_pairs), scores) - f(get_row_indices(S, feature_pairs), scores)) #Marginal gain of current feature pair
    

    S.append(candidate_pairs[candidate_pairs_marginal_gains.index(max(candidate_pairs_marginal_gains))]) #Get max marginal gain, its index and retrieve respective feature pair
    BUDGET = BUDGET - 1

time = time + timer()

print("Final selection: {}".format(S))
print("Execution time: {0:.2f}s".format(time))

#Actual Plotting
outlier_points = [] #Tuple of (best_outliers, other_outliers) for each feature pair; IDS ONLY! MUST RETRIEVE FROM OUTLIER DATAFRAME
#For each selected plot, obtain list of outliers that are best explained by that feature pair (to be drawn in RED)
#Remaining outliers to be drawn in BLUE
for feature_pair in S:
    feature_pair_row_idx = get_row_indices([feature_pair], feature_pairs)[0]
    outliers_max_plot_scores = np.max(scores,axis=0)
    feature_pair_plot_scores = scores[feature_pair_row_idx]
    score_comparison = np.isclose(outliers_max_plot_scores, feature_pair_plot_scores) #Returns boolean array checking if float values are close enough to be considered true #TODO Needs tuning?
    best_outliers_ids = list(map(lambda x: x[0], filter(lambda y: y[1], enumerate(score_comparison.tolist())))) #IDs (in outliers dataframe) of outliers best explained by this feature pair
    remaining_outliers_ids = list(set(range(outlier_df.shape[0])) - set(best_outliers_ids)) #shape property is a fast, safe way to extract number of rows in dataframe (x-shape)
    outlier_points.append((best_outliers_ids, remaining_outliers_ids))


#Create plots and save them to images
#TODO...