'''

Loading a CSV into a table

You've done a great job so far at inserting data into tables! You're now going to learn how to load the contents of a CSV file into a table.

One way to do that would be to read a CSV file line by line, create a dictionary from each line, and then use insert(), like you did in the previous exercise.

But there is a faster way using pandas. You can read a CSV file into a DataFrame using the read_csv() function (this function should be familiar to you, but you can run help(pd.read_csv) in the console to refresh your memory!). Then, you can call the .to_sql() method on the DataFrame to load it into a SQL table in a database. The columns of the DataFrame should match the columns of the SQL table.

.to_sql() has many parameters, but in this exercise we will use the following:

    name is the name of the SQL table (as a string).
    con is the connection to the database that you will use to upload the data.
    if_exists specifies how to behave if the table already exists in the database; possible values are "fail", "replace", and "append".
    index (True or False) specifies whether to write the DataFrame's index as a column.

In this exercise, you will upload the data contained in the census.csv file into an existing table "census". The connection to the database has already been created for you.

Instructions
100 XP

    1   Use pd.read_csv() to load the "census.csv" file into a DataFrame.
            Set the header parameter to None since the file doesn't have a header row.
        Rename the columns of census_df to "state", "sex", age, "pop2000", and "pop2008" to match the columns of the
            "census" table in the database.

    2   Use the .to_sql() method on census_df to append the data to the "census" table in the database using the connection.
        Since "census" already exists in the database, you will need to specify an appropriate value for the if_exists parameter.

'''


# import pandas
import pandas as pd

# read census.csv into a dataframe : census_df
census_df = pd.read_csv("census.csv", header=None)

# rename the columns of the census dataframe
census_df.columns = ['state', 'sex', 'age', 'pop2000', 'pop2008']



# import pandas
import pandas as pd

# read census.csv into a dataframe : census_df
census_df = pd.read_csv("census.csv", header=None)

# rename the columns of the census dataframe
census_df.columns = ['state', 'sex', 'age', 'pop2000', 'pop2008']

# append the data from census_df to the "census" table via connection
census_df.to_sql(name="census", con=connection, if_exists="append", index=False)