'''

Now let's plot a PMF for the age of the respondents in the GSS dataset. The variable 'age' contains
respondents' age in years.

Instructions

    1   Select the 'age' column from the gss DataFrame and store the result in age.     - 35 XP

    2   Make a normalized PMF of age. Store the result in pmf_age.      - 35 XP

    3   Plot pmf_age as a bar chart.        - 30 XP



'''

# Select the age column
age = gss['age']


# Make a PMF of age
pmf_age = Pmf(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()