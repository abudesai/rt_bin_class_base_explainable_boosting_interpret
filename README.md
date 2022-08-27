Explainable Boosting Classifier from Interpret-ML for Binary Classification - Base problem category as per Ready Tensor specifications.

* interpret-ml
* sklearn
* python
* pandas
* numpy
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* binary classification
* glassbox model

This is a Binary Classifier that uses a Explainable Boosting Classifier (EBC) implemented through Interpret-ML.

This EBC is a cyclic gradient boosting Generalized Additive Model that produces results very similar to random forest models while providing global explanations on the impact of each input on the target.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) is conducted by finding the optimal learning rate, number of samples required to be at a leaf node, and tolerance that dictates the smallest delta required to be considered an improvement.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Binary Classifier is written using Python as its programming language. Interpret-ML is used to implement the main algorithm. Scikitlearn is used in the data preprocessing pipeline and model evaluation. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.

