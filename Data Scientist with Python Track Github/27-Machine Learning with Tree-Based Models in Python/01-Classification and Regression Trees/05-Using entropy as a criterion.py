'''

Using entropy as a criterion

In this exercise, you'll train a classification tree on the Wisconsin Breast Cancer dataset using
entropy as an information criterion. You'll do so using all the 30 features in the dataset, which is split into
80% train and 20% test.

X_train as well as the array of labels y_train are available in your workspace.

Instructions
100 XP

    1   Import DecisionTreeClassifier from sklearn.tree.

    2   Instantiate a DecisionTreeClassifier dt_entropy with a maximum depth of 8.

    3   Set the information criterion to 'entropy'.

    4   Fit dt_entropy on the training set.

'''

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)