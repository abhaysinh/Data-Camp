'''
Linking them together!

In the last lesson, you've finished the bulk of the work on your effort to link restaurants and restaurants_new.
You've generated the different pairs of potentially matching rows, searched for exact matches between the cuisine_type
and city columns, but compared for similar strings in the rest_name column. You stored the DataFrame containing the
scores in potential_matches.

Now it's finally time to link both DataFrames. You will do so by first extracting all row indices of restaurants_new
that are matching across the columns mentioned above from potential_matches. Then you will subset restaurants_new on
these indices, then append the non-duplicate values to restaurants. All DataFrames are in your environment, alongside
pandas imported as pd.

Instructions
100 XP

    Isolate instances of potential_matches where the row sum is above or equal to 3 by using the .sum() method.
    Extract the second column index from matches, which represents row indices of matching record from restaurants_new
    by using the .get_level_values() method.
    Subset restaurants_new for rows that are not in matching_indices.
    Append non_dup to restaurants.

'''



# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]

# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)

# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]

# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)