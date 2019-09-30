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

| __Score__ | __Date__ | __Improvement to previous submission__ |
|:-:|:-:|:-:|
| __0.59259__ | 23.09.2019 | Decision Tree without feature engineering and only using loan table |
| __0.61049__ | 23.09.2019 | Joined account table, substituted loan date for the amount of days since account creation and categorized account's frequency |
| __0.56543__ | 24.09.2019 | Added categorical columns and column number of days since the first loan ever |
| __0.62839__ | 24.09.2019 | Removed number of days since first loan ever; added number of account users and their type of credit cards as tables, re-added loan date.
| __0.50000__ | 25.09.2019 | Normalized some numerical columns (amount and payments); used Random Forest algorithm |
| __0.62839__ | 26.09.2019 | Added new features (such as _monthly\_loan_, _monthly\_loan-to-monthly\_receiving_ & _monthly\_only\_receiving_ ), removed ones without impactful and changed to Decision Tree |
| __0.59259__ | 26.09.2019 | Removed _loan\_id_ feature |
| __0.57716__ | 27.09.2019 | Fixed merge of tables in previous submission |
| __0.75370__ | 29.09.2019 | Added transactions table and reworked the flow of the entire project, making it way easier to customize |
| __0.81728__ | 29.09.2019 | Added demographic table |
| __0.84135__ | 30.09.2019 | Removed redundant features, changed join on district_id of account to district_id of client |

## Useful links
* [Principal Component Analysis in 6 steps](https://coolstatsblog.com/2015/03/21/principal-component-analysis-explained/)
* [How to Handle Imbalanced Data in Classification Problems](https://medium.com/james-blogs/handling-imbalanced-data-in-classification-problems-7de598c1059f)
* [Finding Correlation Between Many Variables (Multidimensional Dataset) with Python](https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3)
* [Automated Feature Engineering in Python](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219)
* [A simple guide to creating Predictive Models in Python, Part-1](https://medium.com/datadriveninvestor/a-simple-guide-to-creating-predictive-models-in-python-part-1-8e3ddc3d7008)