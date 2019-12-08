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
#full_df = pd.read_csv("HTRU_2.csv", nrows=1000) #Faster and easier alternative to test (worse results, of course)
full_df = pd.read_csv("HTRU_2.csv")

#Isolate outliers
outlier_df = full_df.loc[full_df['Class'] == 1]

#Remove target column, to not get mixed as a feature
full_df.drop(columns=['Class'], inplace=True)
outlier_df.drop(columns=['Class'], inplace=True)

#Get all available features and combine them 2 by 2
all_features = list(full_df.columns)
feature_pairs = list(ncr(all_features, 2))

#Matrix with scores for all outliers on all feature-pair plots (row = plot, column = outlier)
scores = None
classifier = IF() #Isolation Forest instance used to train and score outliers
for feature_pair in feature_pairs:
    classifier.fit(full_df[list(feature_pair)])
    scores = np.array([classifier.decision_function(outlier_df[list(feature_pair)]).tolist()]) if scores is None \
                else np.append(scores, [classifier.decision_function(outlier_df[list(feature_pair)]).tolist()], axis=0)

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