'''

How can I repeat commands?

One of the biggest advantages of using the shell is that it makes it easy for you to do things over again. If you run some commands, you can then press the up-arrow key to cycle back through them. You can also use the left and right arrow keys and the delete key to edit them. Pressing return will then run the modified command.

Even better, history will print a list of commands you have run recently. Each one is preceded by a serial number to make it easy to re-run particular commands: just type !55 to re-run the 55th command in your history (if you have that many). You can also re-run a command by typing an exclamation mark followed by the command's name, such as !head or !cut, which will re-run the most recent use of that command.

Instructions
100 XP

    1   Run head summer.csv in your home directory (which should fail).

Solution :
head summer.csv

    2   Change directory to seasonal.

Solution :
cd seasonal

    3   Re-run the head command with !head.

Solution :
!head

    4   Use history to look at what you have done.

Solution :
history

    5   Re-run head again using ! followed by a command number.

Solution :
!3
'''