'''
Penalizing highly confident wrong answers

As Peter explained in the video, log loss provides a steep penalty for predictions that are both wrong and confident,
i.e., a high probability is assigned to the incorrect class.

Suppose you have the following 3 examples:

A:y=1,p=0.85

B:y=0,p=0.99

C:y=0,p=0.51

Select the ordering of the examples which corresponds to the lowest to highest log loss scores.
y is an indicator of whether the example was classified correctly. You shouldn't need to crunch any numbers!

Answer the question
50 XP

Possible Answers

    Lowest: A, Middle: B, Highest: C.

    Lowest: C, Middle: A, Highest: B.

    Lowest: A, Middle: C, Highest: B.

    Lowest: B, Middle: A, Highest: C.

Answer :     Lowest: A, Middle: C, Highest: B.
'''