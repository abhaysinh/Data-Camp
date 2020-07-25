'''
Install a specific version of a package (II)

Most commonly, you'll use prefix-notation to specify the package version(s) to install. But conda offers even more powerful comparison operations to narrow versions. For example, if you wish to install either bar-lib versions 1.0, 1.4 or 1.4.1b2, but definitely not version 1.1, 1.2 or 1.3, you could use:

conda install 'bar-lib=1.0|1.4*'

This may seem odd, but you might know, for example, that a bug was introduced in 1.1 that wasn't fixed until 1.4. You would prefer the 1.4 series, but, if it is incompatible with other packages, you can settle for 1.0. Notice we have used single quotes around the version expression in this case because several of the symbols in these more complex patterns have special meanings in terminal shells. It is easiest just to quote them.

With conda you can also use inequality comparisons to select candidate versions (still resolving dependency consistency). Maybe the bug above was fixed in 1.3.5, and you would like either the latest version available (perhaps even 1.5 or 2.0 have come out), but still avoiding versions 1.1 through 1.3.4. You could spell that as:

conda install 'bar-lib>1.3.4,<1.1'

For this exercise, install the latest compatible version of attrs that is later than version 16, but earlier than version 17.3. Which version gets installed?

Instructions
50 XP

Possible Answers

    16.3.0
    17.2.0
    17.2.1
    17.2.3
    17.3.0

Answer :    17.2.0
'''