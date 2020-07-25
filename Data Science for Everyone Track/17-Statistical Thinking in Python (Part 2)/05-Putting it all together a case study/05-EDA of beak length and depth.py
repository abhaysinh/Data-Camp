"""
EDA of beak length and depth

The beak length data are stored as bl_1975 and bl_2012, again with units of millimeters (mm).
You still have the beak depth data stored in bd_1975 and bd_2012. Make scatter plots of beak depth
(y-axis) versus beak length (x-axis) for the 1975 and 2012 specimens.

Instructions
100 XP

    1   Make a scatter plot of the 1975 data. Use the color='blue' keyword argument.
        Also use an alpha=0.5 keyword argument to have transparency in case data points overlap.

    2   Do the same for the 2012 data, but use the color='red' keyword argument.

    3   Add a legend and label the axes.

    4   Show your plot.

"""

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()