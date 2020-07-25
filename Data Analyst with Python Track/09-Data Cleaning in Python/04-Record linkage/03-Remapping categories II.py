'''

Remapping categories II

In the last exercise, you determined that the distance cutoff point for remapping typos of 'american',
'asian', and 'italian' cuisine types stored in the cuisine_type column should be 80.

In this exercise, you're going to put it all together by finding matches with similarity scores equal to or higher
than 80 by using fuzywuzzy.process's extract() function, for each correct cuisine type, and replacing these matches
with it. Remember, when comparing a string with an array of strings using process.extract(), the output is a list of
tuples where each of tuple is as such:

(closest match, similarity score, index of match)

The restaurants DataFrame is in your environment, alongside a categories DataFrame containing the correct cuisine
types in the cuisine_type column.

Instructions
100 XP

    Iterate over each cuisine, in the cuisine_type column of categories.
    For each cuisine, find its similarity to entries in the cuisine_type column of restaurants, while returning all
    possible matches and store them in matches.
    For each possible match in matches equal or higher than 80, find the rows where the cuisine_type in restaurants
    is equal to that possible match.
    Replace that match with the correct cuisine, and print the new unique values of cuisine_type in restaurants.

'''

# For each correct cuisine_type in categories
for cuisine in categories['cuisine_type']:
    # Find matches in cuisine_type of restaurants
    matches = process.extract(cuisine, restaurants['cuisine_type'],
                              limit=restaurants.shape[0])

    # For each possible_match with similarity score >= 80
    for possible_match in matches:
        if possible_match[1] >= 80:
            # Find matching cuisine type
            matching_cuisine = restaurants['cuisine_type'] == possible_match[0]
            restaurants.loc[matching_cuisine, 'cuisine_type'] = cuisine

# Print unique values to confirm mapping
print(restaurants['cuisine_type'].unique())