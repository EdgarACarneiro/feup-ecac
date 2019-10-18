# feup-ecac
Project developed for 'Knowledge Extraction and Machine Learning', a fifth year subject @FEUP. Made in collaboration with @cyrilico.

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

## Submission history

| __Score__ |  __Local Score__ | __Date__ | __Improvement to previous submission__ |
|:-:|:-:|:-:|:-:|
| __0.59259__ | Not recorded | 23.09.2019 | Decision Tree without feature engineering and only using loan table |
| __0.61049__ | Not recorded | 23.09.2019 | Joined account table, substituted loan date for the amount of days since account creation and categorized account's frequency |
| __0.56543__ | Not recorded | 24.09.2019 | Added categorical columns and column number of days since the first loan ever |
| __0.62839__ | Not recorded | 24.09.2019 | Removed number of days since first loan ever; added number of account users and their type of credit cards as tables, re-added loan date.
| __0.50000__ | Not recorded | 25.09.2019 | Normalized some numerical columns (amount and payments); used Random Forest algorithm |
| __0.62839__ | Not recorded | 26.09.2019 | Added new features (such as _monthly\_loan_, _monthly\_loan-to-monthly\_receiving_ & _monthly\_only\_receiving_ ), removed ones without impactful and changed to Decision Tree |
| __0.59259__ | Not recorded | 26.09.2019 | Removed _loan\_id_ feature |
| __0.57716__ | Not recorded | 27.09.2019 | Fixed merge of tables in previous submission |
| __0.75370__ | Not recorded | 29.09.2019 | Added transactions table and reworked the flow of the entire project, making it way easier to customize |
| __0.81728__ | Not recorded | 29.09.2019 | Added demographic table |
| __0.84135__ | Not recorded | 30.09.2019 | Removed redundant features, changed join on district_id of account to district_id of client |
| __0.88148__ | Not recorded | 01.10.2019 | Experimented with grid search hyper parameter running |
| __0.85925__ | Not recorded | 03.10.2019 | Changed Classifying model, after grid searching Decision Tree as it had better performance |
| __0.64197__ | Not recorded | 04.10.2019 | Implemented PCA |
| __0.83580__ | __0.781__ | 04.10.2019 | Increased local score using feature selection |
| __0.89259__ | __0.832430__ | 04.10.2019 | Added class weighting to _RandomForest_ and _GradientBoosting_ |
| __0.85617__ | __0.848035__ | 09.10.2019 | Now considering households and pensions. Fixed numerical imputation not working correctly. |
| __0.82839__ | __0.862035__ | 10.10.2019 | Experimented with under sampling |
| __0.79444__ | __0.840876__ | 10.10.2019 | Added bank demographic data |
| __0.90123__ | __0.842036__ | 11.10.2019 | Heavy feature engineering. Consistent results locally. |
| __0.88333__ | __0.852039__ | 11.10.2019 | Small improvement locally using feature selection and feature engineering. |
| __0.72530__ | __0.841861__ | 12.10.2019 | Heavy feature selection. Removing features without correlation to loan status. |
| __0.770197__ | ____ | 15.10.2019 | Hardcore feature selection. Using only 7 features. |
| __0.85000__ | __0.824199__ | 17.10.2019 | Fixed some local bugs. Heavy feature selection, both automatic and manual. |

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