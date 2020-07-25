'''

Creating two-factor subplots

Let's continue looking at the student_data dataset of students in secondary school.
Here, we want to answer the following question: does a student's first semester grade ("G1") tend to
correlate with their final grade ("G3")?

There are many aspects of a student's life that could result in a higher or lower final grade in the class.
For example, some students receive extra educational support from their school ("schoolsup") or from their family
("famsup"), which could result in higher grades. Let's try to control for these two factors by creating subplots
based on whether the student received extra educational support from their school or family.

Seaborn has been imported as sns and matplotlib.pyplot has been imported as plt.

Instructions

    1 Use relplot() to create a scatter plot with "G1" on the x-axis and "G3" on the y-axis, using the student_data DataFrame. 35 XP
    2 Create column subplots based on whether the student received support from the school ("schoolsup"),
    ordered so that "yes" comes before "no". - 35 XP
    3 Add row subplots based on whether the student received support from the family ("famsup"),
    ordered so that "yes" comes before "no". This will result in subplots based on two factors. - 30 XP



'''

# Create a scatter plot of G1 vs. G3
sns.relplot(x="G1", y="G3",
            data=student_data,
            kind="scatter")

# Show plot
plt.show()

# Adjust to add subplots based on school support
sns.relplot(x="G1", y="G3",
            data=student_data,
            kind="scatter",
            col="schoolsup",
            col_order=["yes", "no"])

# Show plot
plt.show()


# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3",
            data=student_data,
            kind="scatter",
            col="schoolsup",
            col_order=["yes", "no"],
            row="famsup",
            row_order=["yes", "no"])

# Show plot
plt.show()