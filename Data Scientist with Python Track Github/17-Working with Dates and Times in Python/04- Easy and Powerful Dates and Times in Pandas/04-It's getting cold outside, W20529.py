'''
It's getting cold outside, W20529

Washington, D.C. has mild weather overall, but the average high temperature in October (68ºF / 20ºC)
is certainly higher than the average high temperature in December (47ºF / 8ºC). People also travel more in
December, and they work fewer days so they commute less.

How might the weather or the season have affected the length of bike trips?

Instructions

    1   Resample rides to the daily level, based on the Start date column.  - 50 XP
        Plot the .size() of each result.

    2 Since the daily time series is so noisy for this one bike, change the resampling to be monthly. - 50 XP

'''


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to daily, take the size, plot the results
rides.resample('D', on = 'Start date')\
  .size()\
  .plot(ylim = [0, 15])

# Show the results
plt.show()


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to monthly, take the size, plot the results
rides.resample('M', on = 'Start date')\
  .size()\
  .plot(ylim = [0, 150])

# Show the results
plt.show()