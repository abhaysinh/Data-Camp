'''

Make a CDF

In this exercise, you'll make a CDF and use it to determine the fraction of respondents in the
GSS dataset who are OLDER than 30.

The GSS dataset has been preloaded for you into a DataFrame called gss.

As with the Pmf class from the previous lesson, the Cdf class you just saw in the video has been
created for you, and you can access it outside of DataCamp via the empiricaldist library.


Instructions

    1   Select the 'age' column. Store the result in age.   - 50 XP

    2   Compute the CDF of age. Store the result in cdf_age.    - 0 XP

    3   Calculate the CDF of 30.   - 0 XP

    4   Question        - 50 XP

    What fraction of the respondents in the GSS dataset are OLDER than 30?

    Possible Answers

    Approximately 75%
    Approximately 65%
    Approximately 45%
    Approximately 25%

Anser :      Approximately 75%
'''

# Select the age column
age = gss['age']

# Compute the CDF of age
cdf_age = Cdf(age)

# Calculate the CDF of 30
print(cdf_age(30))