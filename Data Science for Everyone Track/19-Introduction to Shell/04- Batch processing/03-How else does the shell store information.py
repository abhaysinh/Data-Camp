'''
How else does the shell store information?

The other kind of variable is called a shell variable, which is like a local variable in a programming language.

To create a shell variable, you simply assign a value to a name:

training=seasonal/summer.csv

without any spaces before or after the = sign. Once you have done this, you can check the variable's value with:

echo $training

seasonal/summer.csv

Instructions
100 XP

    1   Define a variable called testing with the value seasonal/winter.csv.

Solution :

testing=seasonal/winter.csv

    2   Use head -n 1 SOMETHING to get the first line from seasonal/winter.csv
    using the value of the variable testing instead of the name of the file.

Solution : 

# We need to re-set the variable for testing purposes for this exercise
# you should only run "head -n 1 $testing"
testing=seasonal/winter.csv
head -n 1 $testing

'''