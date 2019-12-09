from sklearn.ensemble import IsolationForest as IF
import pandas as pd

full_df = pd.read_csv("HTRU_2.csv")

outlier_df = full_df.loc[full_df['Class'] == 1]
inlier_df = full_df.loc[full_df['Class'] == 0].reset_index().drop(['index'], axis=1)

classes = full_df['Class']
full_df.drop(columns=['Class'], inplace=True)
#inlier_df.drop(columns=['Class'], inplace=True)
outlier_df.drop(columns=['Class'], inplace=True)

classifier = IF()  # Isolation Forest instance used to train and score outliers
classifier.fit(full_df)
scores = classifier.decision_function(outlier_df).tolist()
outlier_df['scores'] = scores
outlier_df = outlier_df.sort_values(by=['scores']).reset_index().drop(['index', 'scores'], axis=1)
outlier_df['Class'] = [1 for i in range(outlier_df.shape[0])]

inlier_df = inlier_df.append(outlier_df.head(32)).reset_index().drop(['index'], axis=1)
inlier_df.to_csv('HTRU_2_filtered.csv', index=False)

