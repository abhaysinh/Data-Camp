'''

Calculating on a pivot table

Pivot tables are filled with summary statistics, but they are only a first step to finding something insightful. Often you'll need to perform further calculations on them. A common thing to do is to find the rows or columns where a highest or lowest value occurs.

Recall from Chapter 1 that you can easily subset a Series or DataFrame to find rows of interest using a logical condition inside of square brackets. For example: series[series > value].

pandas is loaded as pd and the DataFrame temp_by_country_city_vs_year is available.
Instructions
100 XP

    Calculate the mean temperature for each year, assigning to mean_temp_by_year.
    Filter mean_temp_by_year for the year that had the highest mean temperature.
    Calculate the mean temperature for each city (across columns), assigning to mean_temp_by_city.
    Filter mean_temp_by_city for the city that had the lowest mean temperature.

'''

# Get the worldwide mean temp by year
mean_temp_by_year = temp_by_country_city_vs_year.mean()

# Find the year that had the highest mean temp
print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])

# Get the mean temp by city
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis="columns")

# Find the city that had the lowest mean temp
print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])