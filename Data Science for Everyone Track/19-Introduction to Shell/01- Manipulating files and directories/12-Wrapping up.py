'''

Wrapping up

You will often create intermediate files when analyzing data. Rather than storing them in your home directory, you can put them in /tmp, which is where people and programs often keep files they only need briefly. (Note that /tmp is immediately below the root directory /, not below your home directory.) This wrap-up exercise will show you how to do that.

Instructions
100 XP

    1   Use cd to go into /tmp.

Solution :

cd /tmp

    2   List the contents of /tmp without typing a directory name.

Solution :

ls

    3   Make a new directory inside /tmp called scratch.

Solution :
mkdir scratch

    4   Move /home/repl/people/agarwal.txt into /tmp/scratch.
        We suggest you use the ~ shortcut for your home directory and a relative path for the second rather than the absolute path.

Solution :
mv ~/people/agarwal.txt scratch
'''