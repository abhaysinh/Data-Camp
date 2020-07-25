'''
How else can I identify files and directories?

An absolute path is like a latitude and longitude: it has the same value no matter where you are. A relative path, on the other hand, specifies a location starting from where you are: it's like saying "20 kilometers north".

As examples:

    If you are in the directory /home/repl, the relative path seasonal specifies the same directory as the absolute path /home/repl/seasonal.
    If you are in the directory /home/repl/seasonal, the relative path winter.csv specifies the same file as the absolute path /home/repl/seasonal/winter.csv.

The shell decides if a path is absolute or relative by looking at its first character: If it begins with /, it is absolute. If it does not begin with /, it is relative.

Instructions
100 XP

    1   You are in /home/repl. Use ls with a relative path to list the file that has an absolute path of /home/repl/course.txt (and only that file).

Solution

ls course.txt
course.txt

    2   You are in /home/repl. Use ls with a relative path to list the file /home/repl/seasonal/summer.csv (and only that file).

Solution :

ls seasonal/summer.csv
seasonal/summer.csv


    3   You are in /home/repl. Use ls with a relative path to list the contents of the directory /home/repl/people.

Solution:

ls people
agarwal.txt
'''