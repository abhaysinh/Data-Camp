'''
Loading the data

Now it's time to check out the dataset! You'll use pandas (which has been pre-imported as pd) to load your data into
a DataFrame and then do some Exploratory Data Analysis (EDA) of it.

The training data is available as TrainingData.csv. Your first task is to load it into a DataFrame in the IPython
Shell using pd.read_csv() along with the keyword argument index_col=0.

Use methods such as .info(), .head(), and .tail() to explore the budget data and the properties of the features and labels.

Some of the column names correspond to features - descriptions of the budget items - such as the
Job_Title_Description column. The values in this column tell us if a budget item is for a teacher, custodian, or other employee.

Some columns correspond to the budget item labels you will be trying to predict with your model. For example, the
Object_Type column describes whether the budget item is related classroom supplies, salary, travel expenses, etc.

Use df.info() in the IPython Shell to answer the following questions:

    How many rows are there in the training data?
    How many columns are there in the training data?
    How many non-null entries are in the Job_Title_Description column?

Instructions
50 XP

Possible Answers

    25 rows, 1560 columns, 1560 non-null entries in Job_Title_Description.
    25 rows, 1560 columns, 1131 non-null entries in Job_Title_Description.
    1560 rows, 25 columns, 1131 non-null entries in Job_Title_Description.
    1560 rows, 25 columns, 1560 non-null entries in Job_Title_Description.

Answer :  1560 rows, 25 columns, 1131 non-null entries in Job_Title_Description.
'''