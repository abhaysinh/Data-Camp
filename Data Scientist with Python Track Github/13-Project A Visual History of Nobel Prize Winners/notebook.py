
# coding: utf-8

# ## 1. The most Nobel of Prizes
# <p><img style="float: right;margin:5px 20px 5px 1px; max-width:250px" src="https://assets.datacamp.com/production/project_441/img/Nobel_Prize.png"></p>
# <p>The Nobel Prize is perhaps the world's most well known scientific award. Except for the honor, prestige and substantial prize money the recipient also gets a gold medal showing Alfred Nobel (1833 - 1896) who established the prize. Every year it's given to scientists and scholars in the categories chemistry, literature, physics, physiology or medicine, economics, and peace. The first Nobel Prize was handed out in 1901, and at that time the Prize was very Eurocentric and male-focused, but nowadays it's not biased in any way whatsoever. Surely. Right?</p>
# <p>Well, we're going to find out! The Nobel Foundation has made a dataset available of all prize winners from the start of the prize, in 1901, to 2016. Let's load it in and take a look.</p>

# In[78]:


# Loading in required libraries
import pandas as pd
import seaborn as sns
import numpy as np

# Reading in the Nobel Prize data
nobel = pd.read_csv("datasets/nobel.csv")

# Taking a look at the first several winners
nobel.head(n = 6)


# In[79]:


get_ipython().run_cell_magic('nose', '', '\nlast_value = _\n    \ndef test_pandas_loaded():\n    assert pd.__name__ == \'pandas\', \\\n        "pandas should be imported as pd"\n    \ndef test_seaborn_loaded():\n    assert sns.__name__ == \'seaborn\', \\\n        "seaborn should be imported as sns"\n\ndef test_numpy_loaded():\n    assert np.__name__ == \'numpy\', \\\n        "numpy should be imported as np"\n\nimport pandas as pd\n        \ndef test_nobel_correctly_loaded():\n    correct_nobel = pd.read_csv(\'datasets/nobel.csv\')\n    assert correct_nobel.equals(nobel), \\\n        "The variable nobel should contain the data in \'datasets/nobel.csv\'"\n\ndef test_Wilhelm_was_selected():\n    assert "Wilhelm Conrad" in last_value.to_string(), \\\n        "Hmm, it seems you have not displayed at least the first six entries of nobel. A fellow named Wilhelm Conrad Röntgen should be displayed."')


# ## 2. So, who gets the Nobel Prize?
# <p>Just looking at the first couple of prize winners, or Nobel laureates as they are also called, we already see a celebrity: Wilhelm Conrad Röntgen, the guy who discovered X-rays. And actually, we see that all of the winners in 1901 were guys that came from Europe. But that was back in 1901, looking at all winners in the dataset, from 1901 to 2016, which sex and which country is the most commonly represented? </p>
# <p>(For <em>country</em>, we will use the <code>birth_country</code> of the winner, as the <code>organization_country</code> is <code>NaN</code> for all shared Nobel Prizes.)</p>

# In[80]:


# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
display(len(nobel))

# Display the number of prizes won by male and female recipients.
display(nobel['sex'].value_counts())

# Display the number of prizes won by the top 10 nationalities.
nobel['birth_country'].value_counts().head(10)


# In[81]:


get_ipython().run_cell_magic('nose', '', 'last_value = _\n\ncorrect_value = nobel[\'birth_country\'].value_counts().head(10)\n\ndef test_last_value_correct():\n    assert last_value.equals(correct_value), \\\n        "The number of prizes won by the top 10 nationalities doesn\'t seem correct... Maybe check the hint?"')


# ## 3. USA dominance
# <p>Not so surprising perhaps: the most common Nobel laureate between 1901 and 2016 was a man born in the United States of America. But in 1901 all the winners were European. When did the USA start to dominate the Nobel Prize charts?</p>

# In[82]:


# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = nobel['birth_country'] == "United States of America"
nobel['decade'] = (np.floor(nobel['year'] / 10) * 10).astype(int)
prop_usa_winners = nobel.groupby('decade', as_index = False) ['usa_born_winner'].mean()

# Display the proportions of USA born winners per decade
print(prop_usa_winners)


# In[83]:


get_ipython().run_cell_magic('nose', '', '\ndef test_decade_int():\n    assert nobel[\'decade\'].dtype == "int64", \\\n    "Hmm, it looks like the decade column isn\'t calculated correctly. Did you forget to convert it to an integer?"\n\ndef test_correct_prop_usa_winners():\n    correct_prop_usa_winners = nobel.groupby(\'decade\', as_index=False)[\'usa_born_winner\'].mean()\n    assert correct_prop_usa_winners.equals(prop_usa_winners), \\\n        "prop_usa_winners should contain the proportion of usa_born_winner by decade. Don\'t forget to set as_index=False in the groupby() method."')


# ## 4. USA dominance, visualized
# <p>A table is OK, but to <em>see</em> when the USA started to dominate the Nobel charts we need a plot!</p>

# In[84]:


# Setting the plotting theme
sns.set()
# and setting the size of all plots.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [11, 7]

# Plotting USA born winners 
ax = sns.lineplot(x = nobel['decade'], y = nobel['usa_born_winner'])

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter())


# In[85]:


get_ipython().run_cell_magic('nose', '', "\ndef test_y_axis():\n    assert all(ax.get_lines()[0].get_ydata() == prop_usa_winners.usa_born_winner), \\\n    'The plot should be assigned to ax and have usa_born_winner on the y-axis'\n    \ndef test_x_axis():\n    assert all(ax.get_lines()[0].get_xdata() == prop_usa_winners.decade), \\\n    'The plot should be assigned to ax and have decade on the x-axis'")


# ## 5. What is the gender of a typical Nobel Prize winner?
# <p>So the USA became the dominating winner of the Nobel Prize first in the 1930s and had kept the leading position ever since. But one group that was in the lead from the start, and never seems to let go, are <em>men</em>. Maybe it shouldn't come as a shock that there is some imbalance between how many male and female prize winners there are, but how significant is this imbalance? And is it better or worse within specific prize categories like physics, medicine, literature, etc.?</p>

# In[86]:


# Calculating the proportion of female laureates per decade
nobel['female_winner'] = nobel['sex'] == 'Female'
prop_female_winners = nobel.groupby(['decade', 'category'], as_index = False) ['female_winner'].mean()

# Plotting USA born winners with % winners on the y-axis
ax = sns.lineplot(x = 'decade', y = 'female_winner', hue = 'category', data = prop_female_winners)


# In[87]:


get_ipython().run_cell_magic('nose', '', '    \n\ndef test_correct_prop_usa_winners():\n    correct_prop_female_winners = nobel.groupby([\'decade\', \'category\'], as_index=False)[\'female_winner\'].mean()\n    assert correct_prop_female_winners.equals(prop_female_winners), \\\n        "prop_female_winners should contain the proportion of female_winner by decade. Don\'t forget to set as_index=False in the groupby() method."\n\ndef test_y_axis():\n    assert all(pd.Series(ax.get_lines()[0].get_ydata()).isin(prop_female_winners.female_winner)), \\\n    \'The plot should be assigned to ax and have female_winner on the y-axis\'\n    \ndef test_x_axis():\n    assert all(pd.Series(ax.get_lines()[0].get_xdata()).isin(prop_female_winners.decade)), \\\n    \'The plot should be assigned to ax and have decade on the x-axis\'')


# ## 6. The first woman to win the Nobel Prize
# <p>The plot above is a bit messy as the lines are overplotting. But it does show some interesting trends and patterns. Overall the imbalance is pretty large with physics, economics, and chemistry having the largest imbalance. Medicine has a somewhat positive trend, and since the 1990s the literature prize is also now more balanced. The big outlier is the peace prize during the 2010s, but keep in mind that this just covers the years 2010 to 2016.</p>
# <p>Given this imbalance, who was the first woman to receive a Nobel Prize? And in what category?</p>

# In[88]:


# Picking out the first woman to win a Nobel Prize
df1 = nobel[nobel["sex"] == "Female"]
df1.nsmallest(1, "year")


# In[89]:


get_ipython().run_cell_magic('nose', '', '\nlast_value = _\n    \ndef test_Marie_was_selected():\n    assert "Marie Curie" in last_value.to_string(), \\\n        "Hmm, it seems you have not displayed the row of the first woman to win a Nobel Prize, her first name should be Marie."')


# ## 7. Repeat laureates
# <p>For most scientists/writers/activists a Nobel Prize would be the crowning achievement of a long career. But for some people, one is just not enough, and few have gotten it more than once. Who are these lucky few? (Having won no Nobel Prize myself, I'll assume it's just about luck.)</p>

# In[90]:


# Selecting the laureates that have received 2 or more prizes.
nobel.groupby('full_name').filter(lambda x: len(x) >= 2)


# In[91]:


get_ipython().run_cell_magic('nose', '', '\nlast_value = _\n    \ndef test_something():\n    correct_last_value = nobel.groupby(\'full_name\').filter(lambda group: len(group) >= 2)\n    assert correct_last_value.equals(last_value), \\\n        "Did you use groupby followed by the filter method? Did you filter to keep only those with >= 2 prises?"')


# ## 8. How old are you when you get the prize?
# <p>The list of repeat winners contains some illustrious names! We again meet Marie Curie, who got the prize in physics for discovering radiation and in chemistry for isolating radium and polonium. John Bardeen got it twice in physics for transistors and superconductivity, Frederick Sanger got it twice in chemistry, and Linus Carl Pauling got it first in chemistry and later in peace for his work in promoting nuclear disarmament. We also learn that organizations also get the prize as both the Red Cross and the UNHCR have gotten it twice.</p>
# <p>But how old are you generally when you get the prize?</p>

# In[92]:


# Converting birth_date from String to datetime
nobel['birth_date'] = pd.to_datetime(nobel['birth_date'])

# Calculating the age of Nobel Prize winners
nobel['age'] = nobel['year'] - nobel['birth_date'].dt.year

# Plotting the age of Nobel Prize winners
sns.lmplot(x = 'year', y = 'age', data = nobel, lowess = True, aspect = 2, line_kws = {'color': 'black'})


# In[93]:


get_ipython().run_cell_magic('nose', '', '\nax = _\n    \ndef test_birth_date():\n    assert pd.to_datetime(nobel[\'birth_date\']).equals(nobel[\'birth_date\']), \\\n        "Have you converted nobel[\'birth_date\'] using to_datetime?"\n\n    \ndef test_year():\n    assert (nobel[\'year\'] - nobel[\'birth_date\'].dt.year).equals(nobel[\'age\']), \\\n        "Have you caluclated nobel[\'year\'] correctly?"\n\ndef test_plot_data():\n    assert list(ax.data)[0] in ["age", "year"] and list(ax.data)[1] in ["age", "year"], \\\n    \'The plot should show year on the x-axis and age on the y-axis\'\n    \n# Why not this testing code?\n# def test_plot_data():\n#     assert list(ax.data)[0] == "age" and list(ax.data)[1] == "year", \\\n#     \'The plot should show year on the x-axis and age on the y-axis\'')


# ## 9. Age differences between prize categories
# <p>The plot above shows us a lot! We see that people use to be around 55 when they received the price, but nowadays the average is closer to 65. But there is a large spread in the laureates' ages, and while most are 50+, some are very young.</p>
# <p>We also see that the density of points is much high nowadays than in the early 1900s -- nowadays many more of the prizes are shared, and so there are many more winners. We also see that there was a disruption in awarded prizes around the Second World War (1939 - 1945). </p>
# <p>Let's look at age trends within different prize categories.</p>

# In[94]:


# Same plot as above, but separate plots for each type of Nobel Prize
sns.lmplot(x = 'year', y = 'age', data = nobel, row = 'category', lowess = True, aspect = 2, line_kws = {'color': 'black'})


# In[95]:


get_ipython().run_cell_magic('nose', '', '\nax = _\n    \ndef test_plot_data():\n    assert list(ax.data)[0] in ["age", "year", "category"] and \\\n           list(ax.data)[1] in ["age", "year", "category"] and \\\n           list(ax.data)[2] in ["age", "year", "category"], \\\n    \'The plot should show year on the x-axis and age on the y-axis, with one plot row for each category.\'')


# ## 10. Oldest and youngest winners
# <p>More plots with lots of exciting stuff going on! We see that both winners of the chemistry, medicine, and physics prize have gotten older over time. The trend is strongest for physics: the average age used to be below 50, and now it's almost 70. Literature and economics are more stable. We also see that economics is a newer category. But peace shows an opposite trend where winners are getting younger! </p>
# <p>In the peace category we also a winner around 2010 that seems exceptionally young. This begs the questions, who are the oldest and youngest people ever to have won a Nobel Prize?</p>

# In[96]:


# The oldest winner of a Nobel Prize as of 2016
display(nobel.nlargest(1, "age"))

# The youngest winner of a Nobel Prize as of 2016
nobel.nsmallest(1, 'age')


# In[97]:


get_ipython().run_cell_magic('nose', '', '    \nlast_value = _\n    \ndef test_oldest_or_youngest():\n    assert \'Hurwicz\' in last_value.to_string() or \'Yousafzai\' in last_value.to_string(), \\\n        "Have you displayed the row of the oldest winner and the row of the youngest winner?"')


# ## 11. You get a prize!
# <p><img style="float: right;margin:20px 20px 20px 20px; max-width:200px" src="https://assets.datacamp.com/production/project_441/img/paint_nobel_prize.png"></p>
# <p>Hey! You get a prize for making it to the very end of this notebook! It might not be a Nobel Prize, but I made it myself in paint so it should count for something. But don't despair, Leonid Hurwicz was 90 years old when he got his prize, so it might not be too late for you. Who knows.</p>
# <p>Before you leave, what was again the name of the youngest winner ever who in 2014 got the prize for "[her] struggle against the suppression of children and young people and for the right of all children to education"?</p>

# In[98]:


# The name of the youngest winner of the Nobel Prize as of 2016
youngest_winner = 'Malala'


# In[99]:


get_ipython().run_cell_magic('nose', '', '\nimport re\n    \ndef test_right_name():\n    assert re.match("(malala|yousafzai)", youngest_winner.lower()), \\\n        "youngest_winner should be a string. Try writing only the first / given name."')

