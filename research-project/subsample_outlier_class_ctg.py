from sklearn.ensemble import IsolationForest as IF
import pandas as pd

full_df = pd.read_csv("CTG.csv")

full_df.drop(columns=['FileName', 'Date', 'SegFile', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS', 'DR'], inplace=True) #This CLASS is a 10 digit classification system that we're not using
full_df = full_df[full_df['NSP'] != 2] #Suspect class discarded, as described by original link

print(full_df)

outlier_df = full_df.loc[full_df['NSP'] == 3]
inlier_df = full_df.loc[full_df['NSP'] == 1].reset_index().drop(['index'], axis=1)

classes = full_df['NSP']
full_df.drop(columns=['NSP'], inplace=True)
inlier_df.drop(columns=['NSP'], inplace=True)
inlier_df['Class'] = [0 for i in range(inlier_df.shape[0])]
outlier_df.drop(columns=['NSP'], inplace=True)

classifier = IF()  # Isolation Forest instance used to train and score outliers
classifier.fit(full_df)
scores = classifier.decision_function(outlier_df).tolist()
outlier_df['scores'] = scores
outlier_df = outlier_df.sort_values(by=['scores']).reset_index().drop(['index', 'scores'], axis=1)
outlier_df['Class'] = [1 for i in range(outlier_df.shape[0])]

inlier_df = inlier_df.append(outlier_df.head(176)).reset_index().drop(['index'], axis=1) #Subsample to 176 best outliers, as described in original link
inlier_df.to_csv('CTG_Filtered.csv', index=False)

