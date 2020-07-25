'''

Inconsistent categories

In this exercise, you'll be revisiting the airlines DataFrame from the previous lesson.

As a reminder, the DataFrame contains flight metadata such as the airline, the destination,
waiting times as well as answers to key questions regarding cleanliness, safety, and satisfaction on the San Francisco Airport.

In this exercise, you will examine two categorical columns from this DataFrame, dest_region and dest_size respectively,
assess how to address them and make sure that they are cleaned and ready for analysis.
The pandas package has been imported as pd, and the airlines DataFrame is in your environment.


Instructions

    1 Print the unique values in dest_region and dest_size respectively.  - 25 XP

    2 Question           - 25 XP

    From looking at the output, what do you think is the problem with these columns?

    Possible Answers

    The dest_region column has only inconsistent values due to capitalization.
    The dest_region column has inconsistent values due to capitalization and has one value that needs to be remapped.
    The dest_size column has only inconsistent values due to leading and trailing spaces.
    Both 2 and 3 are correct.

Answer :  Both 2 and 3 are correct.


    3 Change the capitalization of all values of dest_region to lowercase.
      Replace the 'eur' with 'europe' in dest_region using the .replace() method.  - 25 XP

    4 Strip white spaces from the dest_size column using the .strip() method.
      Verify that the changes have been into effect by printing the unique values of the columns using .unique() . - 25 XP

'''


# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())


# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})



# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

