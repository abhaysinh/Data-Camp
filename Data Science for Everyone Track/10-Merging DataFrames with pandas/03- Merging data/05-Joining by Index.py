'''


Joining by Index

The DataFrames revenue and managers are displayed in the IPython Shell. Here, they are indexed by 'branch_id'.

Choose the function call below that will join the DataFrames on their indexes and return 5 rows with index labels [10, 20, 30, 31, 47]. Explore each of them in the IPython Shell to get a better understanding of their functionality.

                 city state  revenue
branch_id
10              Austin    TX      100
20              Denver    CO       83
30         Springfield    IL        4
47           Mendocino    CA      200

                branch state   manager
branch_id
10              Austin    TX  Charlers
20              Denver    CO      Joel
47           Mendocino    CA     Brett
31         Springfield    MO     Sally

Possible Answers

    pd.merge(revenue, managers, on='branch_id').
    pd.merge(managers, revenue, how='left').
    revenue.join(managers, lsuffix='_rev', rsuffix='_mng', how='outer').
    managers.join(revenue, lsuffix='_mgn', rsuffix='_rev', how='left').

    Answer: revenue.join(managers, lsuffix='_rev', rsuffix='_mng', how='outer').

'''