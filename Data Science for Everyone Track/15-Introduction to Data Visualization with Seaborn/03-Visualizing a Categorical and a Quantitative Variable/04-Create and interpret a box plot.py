'''

Create and interpret a box plot

Let's continue using the student_data dataset. In an earlier exercise, we explored the relationship between
studying and final grade by using a bar plot to compare the average final grade ("G3") among students in different
categories of "study_time".

In this exercise, we'll try using a box plot look at this relationship instead. As a reminder, to create a box plot
you'll need to use the catplot() function and specify the name of the categorical variable to put on the x-axis
(x=____), the name of the quantitative variable to summarize on the y-axis (y=____), the Pandas DataFrame to use
(data=____), and the type of plot (kind="box").

We have already imported matplotlib.pyplot as plt and seaborn as sns.

Instructions

    1 Use sns.catplot() and the student_data DataFrame to create a box plot with "study_time" on the x-axis
    and "G3" on the y-axis. Set the ordering of the categories to study_time_order. 50 XP

    2 Question 50 XP

Which of the following is a correct interpretation of this box plot?
Possible Answers

    The 75th percentile of grades is highest among students who study more than 10 hours a week.
    There are no outliers plotted for these box plots.
    The 5th percentile of grades among students studying less than 2 hours is 5.0.
    The median grade among students studying less than 2 hours is 10.0.

Answer : The median grade among students studying less than 2 hours is 10.0.

'''

# Specify the category ordering
study_time_order = ["<2 hours", "2 to 5 hours",
                    "5 to 10 hours", ">10 hours"]

# Create a box plot and set the order of the categories
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="box",
            order=study_time_order)

# Show plot
plt.show()


