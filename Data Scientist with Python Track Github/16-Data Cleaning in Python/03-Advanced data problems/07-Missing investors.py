'''

Missing investors

Dealing with missing data is one of the most common tasks in data science.
There are a variety of types of missingness, as well as a variety of types of solutions to missing data.

You just received a new version of the banking DataFrame containing data on the amount held and invested for
new and existing customers. However, there are rows with missing inv_amount values.

You know for a fact that most customers below 25 do not have investment accounts yet, and suspect it could
be driving the missingness. The pandas, missingno and matplotlib.pyplot packages have been imported as pd, msno
and plt respectively. The banking DataFrame is in your environment.


Instructions

    1 Print the number of missing values by column in the banking DataFrame.
      Plot and show the missingness matrix of banking with the msno.matrix() function.  - 25 XP

    2 Isolate the values of banking missing values of inv_amount into missing_investors
    and with non-missing inv_amount values into investors.   - 25 XP

    3 Question  - 25 XP

    Now that you've isolated banking into investors and missing_investors, use the .describe() method on both of
    these DataFrames in the console to understand whether there are structural differences between them.
    What do you think is going on?

    Possible Answers

    The data is missing completely at random and there are no drivers behind the missingness.
    The inv_amount is missing only for young customers, since the average age in missing_investors is 22 and the maximum age is 25.
    The inv_amount is missing only for old customers, since the average age in missing_investors is 42 and the maximum age is 59.

Answer :  The inv_amount is missing only for young customers, since the average age in missing_investors is 22 and the maximum age is 25.

    4 Sort the banking DataFrame by the age column and plot the missingness matrix of banking_sorted.  - 25 XP

'''


# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()


# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]



# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]

# Sort banking by age and visualize
banking_sorted = banking.sort_values(by = 'age')
msno.matrix(banking_sorted)
plt.show()