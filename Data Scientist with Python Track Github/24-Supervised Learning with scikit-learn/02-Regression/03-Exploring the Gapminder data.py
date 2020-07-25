'''
Exploring the Gapminder data

As always, it is important to explore your data before building models. On the right, we have constructed
a heatmap showing the correlation between the different features of the Gapminder dataset, which has been
pre-loaded into a DataFrame as df and is available for exploration in the IPython Shell. Cells that are in
green show positive correlation, while cells that are in red show negative correlation. Take a moment to explore this:
Which features are positively correlated with life, and which ones are negatively correlated? Does this match your intuition?

Then, in the IPython Shell, explore the DataFrame using pandas methods such as .info(), .describe(), .head().

In case you are curious, the heatmap was generated using Seaborn's heatmap function and the following
line of code, where df.corr() computes the pairwise correlation between columns:

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

Once you have a feel for the data, consider the statements below and select the one that is not true.
After this, Hugo will explain the mechanics of linear regression in the next video and you will be on
your way building regression models!

Instructions
50 XP

Possible Answers

    The DataFrame has 139 samples (or rows) and 9 columns.
    life and fertility are negatively correlated.
    The mean of life is 69.602878.
    fertility is of type int64.
    GDP and life are positively correlated.

Answer : fertility is of type int64.
'''