from sklearn.ensemble import IsolationForest as IF
import pandas as pd
from itertools import combinations as ncr
import numpy as np

#Number of plots to choose
BUDGET = 3

#Load dataset
full_df = pd.read_csv("HTRU_2.csv", nrows=100) #Faster and easier alternative to test (worse results, of course)
#full_df = pd.read_csv("HTRU_2.csv")

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

# clf = IF()
# clf.fit(df)
# #print(clf.decision_function(df))
# y_pred_train = clf.predict(df)
# for idx, comp in list(enumerate(zip(y_pred_train, real_outliers)))[:100]:
#     if comp[0] != comp[1]:
#         print("{} different: expected {}, but got {}".format(idx, comp[1], comp[0]))