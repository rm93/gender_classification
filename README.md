# gender_classification

<h3>Overview</h3>

The code uses the [scikit-learn](http://scikit-learn.org/) machine learning library to train six 
diverent algorithms on a small dataset of body metrics (height, weight and shoe_size) labeled male or female. Then we can predict
the gender of someone give a novel set of body metrics.

This is my first machine learning python script

<h3>Dependencies</h3>

* Pickle ```pip install pickle```
* [Scikit-learn](http://scikit-learn.org/) ```pip install -U scikit-learn```

Install missing dependencies using pip

<h3>Usage</h3>

Once you have your dependencies installed via pip, run the script in terminal:

1. Run ```python learning.py``` to train the model.
2. Once trained you can test the model via ```python predictor.py```. It will output either male or female.
