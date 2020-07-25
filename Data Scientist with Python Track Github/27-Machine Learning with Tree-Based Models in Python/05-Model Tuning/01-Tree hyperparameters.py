'''

ree hyperparameters

In the following exercises you'll revisit the Indian Liver Patient dataset (https://www.kaggle.com/uciml/indian-liver-patient-records)
which was introduced in a previous chapter.

Your task is to tune the hyperparameters of a classification tree. Given that this dataset is imbalanced,
you'll be using the ROC AUC score as a metric instead of accuracy.

We have instantiated a DecisionTreeClassifier and assigned to dt with sklearn's default hyperparameters.
You can inspect the hyperparameters of dt in your console.

Which of the following is not a hyperparameter of dt?

Instructions
50 XP

Possible Answers

    min_impurity_decrease
    min_weight_fraction_leaf
    min_features
    splitter

Answer : min_features
'''