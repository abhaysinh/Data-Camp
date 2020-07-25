'''

Appending Series with nonunique Indices

The Series bronze and silver, which have been printed in the IPython Shell, represent the 5 countries that won the most bronze and silver Olympic medals respectively between 1896 & 2008. The Indexes of both Series are called Country and the values are the corresponding number of medals won.

If you were to run the command combined = bronze.append(silver), how many rows would combined have? And how many rows would combined.loc['United States'] return? Find out for yourself by running these commands in the IPython Shell.

bronze
Country
United States     1052.0
Soviet Union       584.0
United Kingdom     505.0
France             475.0
Germany            454.0
Name: Total, dtype: float64

silver
Country
United States     1195.0
Soviet Union       627.0
United Kingdom     591.0
France             461.0
Italy              394.0
Name: Total, dtype: float64


Possible Answers

    combined has 5 rows and combined.loc['United States'] is empty (0 rows).
    combined has 10 rows and combined.loc['United States'] has 2 rows.
    combined has 6 rows and combined.loc['United States'] has 1 row.
    combined has 5 rows and combined.loc['United States'] has 2 rows.

    Answer:  combined has 10 rows and combined.loc['United States'] has 2 rows.
'''