'''
Install a conda package (II)

Installing a package is largely a matter of listing the name(s) of packages to install after the command conda install. But there is more to it behind the scenes. The versions of packages to install (along with all their dependencies) must be compatible with all versions of other software currently installed. Often this "satisfiability" constraint depends on choosing a package version compatible with a particular version of Python that is installed. Conda is special among "package managers" in that it always guarantees this consistency; you will see the phrase "Solving environment..." during installation to indicate this computation.

For example, you may simply instruct conda to install foo-lib. The tool first determines which operating system you are running, and then narrows the match to candidates made for this platform. Then, conda determines the version of Python on the system (say 3.7), and chooses the package version for -py37. But, beyond those simple limits, all dependencies are checked.

Suppose foo-lib is available in versions 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1 (for your platform and Python version). As a first goal, conda attempts to choose the latest version of foo-lib. However, maybe foo-lib depends on bar-lib, which itself is available in various versions (say 1 through 20 in its versioning scheme). It might be that foo-lib 3.1 is compatible with bar-lib versions 17, 18, and 19; but blob-lib (which is already installed) is compatible only with versions of bar-lib less than 17. Therefore, conda would examine the compatibility of foo-lib 3.0 as a fallback. In this hypothetical, foo-lib 3.0 is compatible with bar-lib 16, so that version is chosen (bar-lib is also updated to the latest compatible version 16 in the same command if an earlier version is currently installed).

Visually (octagons mark chosen versions):

Instructions
100 XP

    Install the package cytoolz using conda. (Press y when asked to proceed.)

Solution :

# The --yes flag is added here to avoid interactivity when
# the course builds. It isn't required to complete the
# exercise.
conda install cytoolz --yes
'''