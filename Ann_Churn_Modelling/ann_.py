import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
# from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9, input_dim=11))
classifier.add(Dropout(rate=0.1))
# Adding the second hidden layer
classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
classifier.add(Dropout(rate=0.1))
# Building final layer
classifier.add(Dense(activation="sigmoid", kernel_initializer="random_uniform", units=1))
# Compiling the ANN
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=32, epochs=500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_testset = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

# Pre-Processing
hw = pd.read_csv('test.csv')
hw = hw.iloc[:, :].values
hw[:, 1] = labelencoder_X_1.fit_transform(hw[:, 1])
hw[:, 2] = labelencoder_X_2.fit_transform(hw[:, 2])
hw = onehotencoder.fit_transform(hw).toarray()
hw = hw[0:1, 1:]
hw = sc.transform(hw)
# Predicting the outcome
hw_pred = classifier.predict(hw)
hw_pred = hw_pred > 0.5

# Evaluating the Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9, input_dim=11))
    classifier.add(Dropout(rate=0.1))
    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(activation="sigmoid", kernel_initializer="random_uniform", units=1))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean_accuracies = accuracies.mean()
variance_accuracies = accuracies.std()

# GRID Search Parameter Tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9, input_dim=11))  # Adding the second hidden layer
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dense(activation="relu", kernel_initializer="random_uniform", units=9))
    classifier.add(Dense(activation="sigmoid", kernel_initializer="random_uniform", units=1))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
