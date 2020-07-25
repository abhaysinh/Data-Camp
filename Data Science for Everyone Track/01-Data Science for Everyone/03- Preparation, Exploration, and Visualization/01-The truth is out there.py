'''
The truth is out there

Yesterday night, you were stargazing with a friend and you saw something strange in the sky. You decide to look into UFO (Unidentified Flying Object) reports to see if anyone has seen something similar in your area. Luckily, you find a UFO reports dataset, but before analyzing it, you need to prepare the data! Here is a snapshot of the first 5 rows.

Flowers

Which of the following statements is true?

Answer the question
50 XP

Possible Answers

    The last row is missing a value for city_latitude, you should drop it.

    The third row is full of NaN values. A UFO sighting event has to have a place, a date, and a description (the shape of the object). This row does not have these information, you should drop it.

    The third row is full of NaN values, which is unusual. You should try to understand why that is the case before taking the decision to drop or keep it.

    These five observations report seeing aliens, so what you and your friend saw was probably aliens too.

Answer : The third row is full of NaN values, which is unusual. You should try to understand why that is the case before taking the decision to drop or keep it.

'''