'''

Install a specific version of a package (I)

Sometimes there are reasons why you need to use a specific version of a package, not necessarily simply the latest version compatible with your other installed software. You may have scripts written that depend on particular older APIs, or you may have received code written by colleagues who used specific versions and you want to guarantee replication of the same behavior. Likewise, you may be writing code that you intend to pass to other users who you know to be using specific package versions on their systems (perhaps as a company standard, for example).

conda allows you to install software versions in several flexible ways. Your most common pattern will probably be prefix notation, using semantic versioning. For example, you might want a MAJOR and MINOR version, but want conda to select the most up-to-date PATCH version within that series. You could spell that as:

conda install foo-lib=12.3

Or similarly, you may want a particular major version, and prefer conda to select the latest compatible MINOR version as well as PATCH level. You could spell that as:

conda install foo-lib=13

If you want to narrow the installation down to an exact PATCH level, you can specify that as well with:

conda install foo-lib=14.3.2

Keep in mind that relaxing constraints may allow for satisfying multiple dependencies among installed software. Occasionally you will try to install some software version that is simply inconsistent with other software installed, and conda will warn you about that rather than install anything.

Instructions
100 XP

        Install the module attrs in the specific MAJOR and MINOR version 17.3.

Solution:

# The --yes flag is added here to avoid interactivity when
# the course builds. It isn't required to complete the
# exercise.
conda install attrs=17.3 --yes
'''