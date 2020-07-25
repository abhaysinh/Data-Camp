'''

Train an RF regressor

In the following exercises you'll predict bike rental demand in the Capital Bikeshare program in Washington,
D.C using historical weather data from the Bike Sharing Demand dataset (https://www.kaggle.com/c/bike-sharing-demand)
available through Kaggle. For this purpose, you will be using the random forests algorithm.
As a first step, you'll define a random forests regressor and fit it to the training set.

The dataset is processed for you and split into 80% train and 20% test. The features matrix X_train and the
array y_train are available in your workspace.

Instructions
100 XP

    1   Import RandomForestRegressor from sklearn.ensemble.

    2   Instantiate a RandomForestRegressor called rf consisting of 25 trees.

    3   Fit rf to the training set.

'''

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
                           random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)