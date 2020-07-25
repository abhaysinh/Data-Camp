'''

How can I rename files?

mv can also be used to rename files. If you run:

mv course.txt old-course.txt

then the file course.txt in the current working directory is "moved" to the file old-course.txt. This is different from the way file browsers work, but is often handy.

One warning: just like cp, mv will overwrite existing files. If, for example, you already have a file called old-course.txt, then the command shown above will replace it with whatever is in course.txt.

Instructions
100 XP

    1   Go into the seasonal directory.

Solution :

cd seasonal

    2   Rename the file winter.csv to be winter.csv.bck.

Solution :
mv winter.csv winter.csv.bck


    3   Run ls to check that everything has worked.

Solution :
ls
'''