# Artificial Neural Network to profile new loan applications
* Developed with Tensorflow and Scikit-Learn
* The model predicts whether a prospective customer will default or be able to repay their loan in full.
* All model pipelines are saved, stored, and reloaded to create a nice ML Pipeline
* Data from Kaggle

## Information about the ANN
* In depth data exploration with pandas (bulk of the project)
* Imputing missing data with pandas (not with Scikit-learn Imputers)
* Encoding categorical features with pandas (not with Scikit-learn OneHotEncoder and ColumnTransformer)
* Data training and test split done with Scikit-learn
* Feature scaling is done with Sckik-learn MinMaxScaler
* ANN model built with tensorflow
* ANN hidden layers each contain dropout layers
* Keras EarlyStop callback included as argument in model fitting
* Model loss and validation loss history is visualised to illustrate the early stop which prevents overfitting 
* Model developed with Tensorflow.Keras.Sequentia
* ML Pipeline: Scalers, Trained Model, Final DataFrame and Model history all stored in either hdf5 or csv formats, and can be reloaded each time.


## Packages
* See requirements.txt for all packages and dependencies used in my virtual environment.