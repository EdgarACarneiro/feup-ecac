# feup-ecac
Project developed for 'Knowledge Extraction and Machine Learning', a fifth year subject @FEUP. Made in collaboration with [@cyrilico](https://github.com/cyrilico).

A Summary of the theoretical material is available [here](https://github.com/EdgarACarneiro/feup-ecac/blob/master/Summary.md).

Folder `/research-project` contains materials that were necessary to the develop the ECAC research project (2nd project).

## Project Grade

| Component | Grade |
|:-|:-|
| __Project__ | 20 |
| __Classification__ | 17 | 

## Usage

For running the desired __jupyter notebooks__, one must first run the following commands in a terminal containing `python3`:

* In Mac/ Linux
```shell
python3  -m venv venv
. venv/bin/activate
pip install _U -r requirements.txt
jupyter notebook
```

* Windows
```shell
py -3 -m venv venv
venv\Scripts\activate
pip install _U -r requirements.txt
jupyter notebook
```

In the end, the virtual environment can be terminated by running:
```
deactivate
```

In the jupyter notebook web page, open the [__pre_processing.ipynb__](https://github.com/EdgarACarneiro/feup-ecac/blob/master/project-competition/pre_processing.ipynb) file first and simply run all cells.

Thereafter, open the [__prediction.ipynb__](https://github.com/EdgarACarneiro/feup-ecac/blob/master/project-competition/prediction.ipynb) file. Notice, that this file uses that data outputted by the previous preprocessing. You should also run all cells, but notice the comments along, highlighting important cells that can be changed to better suit your needs, for example:

```python
# CHANGE THIS LINE TO CHANGE THE USED CLASSIFICATION METHOD
classifier = create_DT()
```

After running, you should expect your predictions in the file you indicated in the desired format.

## Final submission

Final presentation slides available [here](https://github.com/EdgarACarneiro/feup-ecac/blob/master/docs/ECAC%20Presentation%20-%20T1%20G10.pdf).

Final leaderboards available [here](https://www.kaggle.com/c/to-loan-or-not-to-loan-4/leaderboard) - placed :nine:.

## Submission history

* :heavy_exclamation_mark: : Submissions selected for competition scoring. Notice that we did not have access to the private score when choosing the two submissions.

| __Public Score__ | __Private Score__ | __Local Score__ | __Date__ | __Improvement to previous submission__ |
|:-:|:-:|:-:|:-:|:-|
| __0.59259__ | __0.57160__ | Not recorded | 23.09.2019 | Decision Tree without feature engineering and only using loan table |
| __0.61049__ | __0.59876__ | Not recorded | 23.09.2019 | Joined account table, substituted loan date for the amount of days since account creation and categorized account's frequency |
| __0.56543__ | __0.61728__ | Not recorded | 24.09.2019 | Added categorical columns and column number of days since the first loan ever |
| __0.62839__ | __0.65864__ | Not recorded | 24.09.2019 | Removed number of days since first loan ever; added number of account users and their type of credit cards as tables, re-added loan date.
| __0.50000__ | __0.50000__ | Not recorded | 25.09.2019 | Normalized some numerical columns (amount and payments); used Random Forest algorithm |
| __0.62839__ | __0.58888__ | Not recorded | 26.09.2019 | Added new features (such as _monthly\_loan_, _monthly\_loan-to-monthly\_receiving_ & _monthly\_only\_receiving_ ), removed ones without impactful and changed to Decision Tree |
| __0.59259__ | __0.63209__ | Not recorded | 26.09.2019 | Removed _loan\_id_ feature |
| __0.57716__ | __0.60802__ | Not recorded | 27.09.2019 | Fixed merge of tables in previous submission |
| __0.75370__ | __0.75308__ | Not recorded | 29.09.2019 | Added transactions table and reworked the flow of the entire project, making it way easier to customize |
| __0.81728__ | __0.75679__ | Not recorded | 29.09.2019 | Added demographic table |
| __0.84135__ | __0.77716__ | Not recorded | 30.09.2019 | Removed redundant features, changed join on district_id of account to district_id of client |
| __0.88148__ | __0.68148__ | Not recorded | 01.10.2019 | Experimented with grid search hyper parameter running |
| __0.85925__ | __0.73518__ | Not recorded | 03.10.2019 | Changed Classifying model, after grid searching Decision Tree as it had better performance |
| __0.64197__ | __0.59876__ | Not recorded | 04.10.2019 | Implemented PCA |
| __0.83580__ | __0.80555__ | __0.781090__ | 04.10.2019 | Increased local score using feature selection |
| __0.89259__ | __0.75555__ | __0.832430__ | 04.10.2019 | Added class weighting to _RandomForest_ and _GradientBoosting_ |
| __0.85617__ | __0.73765__ | __0.848035__ | 09.10.2019 | Now considering households and pensions. Fixed numerical imputation not working correctly. |
| __0.82839__ | __0.72530__ | __0.862035__ | 10.10.2019 | Experimented with under sampling |
| __0.79444__ | __0.64012__ | __0.840876__ | 10.10.2019 | Added bank demographic data |
| :heavy_exclamation_mark: __0.90123__ | __0.79506__ | __0.842036__ | 11.10.2019 | Heavy feature engineering. Consistent results locally. |
| __0.88333__ | __0.81666__ | __0.852039__ | 11.10.2019 | Small improvement locally using feature selection and feature engineering. |
| __0.72530__ | __0.71913__ | __0.841861__ | 12.10.2019 | Heavy feature selection. Removing features without correlation to loan status. |
| __0.77020__ | __0.73333__ | Not recorded | 15.10.2019 | Hardcore feature selection. Using only 7 features. |
| __0.85000__ | __0.81049__ | __0.824199__ | 17.10.2019 | Fixed some local bugs. Heavy feature selection, both automatic and manual. |
| __0.79753__ | __0.68827__ | __0.828777__ | 18.10.2019 | Very consistent results. S'more feature engineering and selection. |
| __0.77160__ | __0.75617__ | __0.799563__ | 19.10.2019 | Decision Tree of depth 2. Constant AUC of 80%, probably small error interval. |
| __0.78353__ | __0.68353__ | __0.937524__ | 21.10.2019 | Applied backward elimination. Using LinearRegression. Constant local score. |
| __0.70432__ | __0.58271__ | __0.860821__ | 21.10.2019 | Feature selection using backward elimination and RFE on LogisticRegression |
| __0.71913__ | __0.83395__ | __0.845231__ | 24.10.2019 |  Using most consistent local with SMOTETek sampling and Gradient Boosting. |
| :heavy_exclamation_mark: __0.85864__ | __0.74012__ | __0.867982__| 24.10.2019 | Best local scoring setup. |
| __0.83209__ | __0.78641__ | __0.864521__ | 25.10.2019 | Random Forest with SMOTETEEN and Filter Method as feature selection. Locally consistent. |
| __0.74074__ | __0.79506__ | __0.850971__ | 25.10.2019 | Best local Decision Tree, with SMOTETEEN and Filter Method as feature selection. Likely to overfit. | 

## Useful links
* [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
* [Principal Component Analysis in 6 steps](https://coolstatsblog.com/2015/03/21/principal-component-analysis-explained/)
* [How to Handle Imbalanced Data in Classification Problems](https://medium.com/james-blogs/handling-imbalanced-data-in-classification-problems-7de598c1059f)
* [Finding Correlation Between Many Variables (Multidimensional Dataset) with Python](https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3)
* [Automated Feature Engineering in Python](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219)
* [A simple guide to creating Predictive Models in Python, Part-1](https://medium.com/datadriveninvestor/a-simple-guide-to-creating-predictive-models-in-python-part-1-8e3ddc3d7008)
* [Automated Machine Learning Hyperparameter Tuning in Python](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a)
* [Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)
* [Feature Selection and Dimensionality Reduction](https://towardsdatascience.com/feature-selection-and-dimensionality-reduction-f488d1a035de)
* [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* [Understanding PCA (Principal Component Analysis) with Python](https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21)
* [Loan Default Prediction and Identification of Interesting Relations between Attributes of Peer-to-Peer Loan Applications](https://www.researchgate.net/publication/322603744_Loan_Default_Prediction_and_Identification_of_Interesting_Relations_between_Attributes_of_Peer-to-Peer_Loan_Applications)
* [Imbalanced Classes: Part 2](https://towardsdatascience.com/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-part-2-cf371500d1b3)
* [Feature Selection with sklearn and Pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
* [Feature Selection Using Random forest](https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f)
