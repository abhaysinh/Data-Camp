'''
Update a conda package

Closely related to installing a particular version of a conda package is updating the installed version to the latest version possible that remains compatible with other installed software. conda will determine if it is possible to update dependencies of the package(s) you are directly updating, and do so if resolvable. At times, the single specified package will be updated exclusively since the current dependencies are correct for the new version. Obviously, at other times updating will do nothing because you are already at the latest version possible.

The command conda update PKGNAME is used to perform updates. Update is somewhat less "aggressive" than install in the sense that installing a specific (later) version will revise the versions in the dependency tree to a greater extent than an update. Often update will simply choose a later PATCH version even though potentially a later MAJOR or MINOR version could be made compatible with other installed packages.

Note that this conda command, as well as most others allow specification of multiple packages on the same line. For example, you might use:

conda update foo bar blob

To bring all of foo, bar, and blob up to the latest compatible versions mutually satisfiable.

Instructions
100 XP

    The package pandas is installed in the current image, but it's not the most recent version. Update it.

Solution :

# The --yes flag is added here to avoid interactivity when
# the course builds. It isn't required to complete the
# exercise.
conda update pandas --yes

'''