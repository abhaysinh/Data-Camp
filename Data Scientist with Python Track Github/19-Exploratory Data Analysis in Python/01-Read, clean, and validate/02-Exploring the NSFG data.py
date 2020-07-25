'''

Exploring the NSFG data

To get the number of rows and columns in a DataFrame, you can read its shape attribute.

To get the column names, you can read the columns attribute. The result is an Index, which is a Pandas
data structure that is similar to a list. Let's begin exploring the NSFG data! It has been pre-loaded for you
into a DataFrame called nsfg.

Instructions

    1   Calculate the number of rows and columns in the DataFrame nsfg.  - 25 XP

    2   Display the names of the columns in nsfg.   - 25 XP

    3   Select the column 'birthwgt_oz1' and assign it to a new variable called ounces.   - 25 XP

    4   Display the first 5 elements of ounces.         - 25 XP

'''


# Display the number of rows and columns
print(nsfg.shape)


# Display the names of the columns
print(nsfg.columns)


# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']


# Print the first 5 elements of ounces
print(ounces.head())