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

# Submission history

| __Score__ | __Date__ | __Improvement to previous submission__ |
|:-:|:-:|:-:|
| __0.59259__ | 23.09.2019 | Decision Tree without feature engineering and only using loan table |
