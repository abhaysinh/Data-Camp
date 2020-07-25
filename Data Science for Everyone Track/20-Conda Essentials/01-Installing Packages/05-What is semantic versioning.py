'''
What is semantic versioning?

Most Conda packages use a system called semantic versioning to identify distinct versions of a software package unambiguously. Version labels are usually chosen by the project authors, not necessarily the same people who bundle the project as a Conda package. There is no technical requirement that a project author's version label coincides with a Conda package version label, but the convention of doing so is almost always followed. Semantic version labels can be compared lexicographically and make it easy to determine which of two versions is the later version.

Under semantic versioning, software is labeled with a three-part version identifier of the form MAJOR.MINOR.PATCH; the label components are non-negative integers separated by periods. Assuming all software starts at version 0.0.0, the MAJOR version number is increased when significant new functionality is introduced (often with corresponding API changes). Increases in the MINOR version number generally reflect improvements (e.g., new features) that avoid backward-incompatible API changes. For instance, adding an optional argument to a function API (in a way that allows old code to run unchanged) is a change worthy of increasing the MINOR version number. An increment to the PATCH version number is appropriate mostly for bug fixes that preserve the same MAJOR and MINOR revision numbers. Software patches do not typically introduce new features or change APIs at all (except sometimes to address security issues).

Many command-line tools display their version identifier by running tool --version. This information can sometimes be displayed or documented in other ways. For example, suppose on some system, a certain version of Python is installed, and you inquire about it like this:

python -c "import sys; sys.version"
'1.0.1 (Mar 26 2014)'

Looking at the output above, which statement below accurately characterizes the semantic versioning of this installed Python?

Answer the question
50 XP

Possible Answers

    The MAJOR version is 0, the MINOR version is 1.

    The MAJOR version is 1, the PATCH is Mar 26 2014.

    The MAJOR version is 1, the PATCH is 1.

    The MAJOR version is 1.0, the PATCH is 1.


    This ancient version of Python did not use semantic versioning.

Answer :     The MAJOR version is 1, the PATCH is 1.
'''