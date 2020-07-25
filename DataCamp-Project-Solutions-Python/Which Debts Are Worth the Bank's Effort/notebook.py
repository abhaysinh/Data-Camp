#!/usr/bin/env python
# coding: utf-8

# ## 1. Regression discontinuity: banking recovery
# <p>After a debt has been legally declared "uncollectable" by a bank, the account is considered "charged-off." But that doesn't mean the bank <strong><em>walks away</em></strong> from the debt. They still want to collect some of the money they are owed. The bank will score the account to assess the expected recovery amount, that is, the expected amount that the bank may be able to receive from the customer in the future. This amount is a function of the probability of the customer paying, the total debt, and other factors that impact the ability and willingness to pay.</p>
# <p>The bank has implemented different recovery strategies at different thresholds (\$1000, \$2000, etc.) where the greater the expected recovery amount, the more effort the bank puts into contacting the customer. For low recovery amounts (Level 0), the bank just adds the customer's contact information to their automatic dialer and emailing system. For higher recovery strategies, the bank incurs more costs as they leverage human resources in more efforts to obtain payments. Each additional level of recovery strategy requires an additional \$50 per customer so that customers in the Recovery Strategy Level 1 cost the company \$50 more than those in Level 0. Customers in Level 2 cost \$50 more than those in Level 1, etc. </p>
# <p><strong>The big question</strong>: does the extra amount that is recovered at the higher strategy level exceed the extra \$50 in costs? In other words, was there a jump (also called a "discontinuity") of more than \$50 in the amount recovered at the higher strategy level? We'll find out in this notebook.</p>
# <p>![Regression discontinuity graph](https://assets.datacamp.com/production/project_504/img/Regression Discontinuity graph.png)</p>
# <p>First, we'll load the banking dataset and look at the first few rows of data. This lets us understand the dataset itself and begin thinking about how to analyze the data.</p>

# In[2]:


# Import modules
import pandas as pd
import numpy as np

# Read in dataset
df = pd.read_csv("datasets/bank_data.csv")

# Print the first few rows of the DataFrame
df.head()


# In[3]:


get_ipython().run_cell_magic('nose', '', '\nfirst_value = _\n    \ndef test_pandas_loaded():\n    assert pd.__name__ == \'pandas\', \\\n        "pandas should be imported as pd."\n\ndef test_numpy_loaded():\n    assert np.__name__ == \'numpy\', \\\n        "numpy should be imported as np."\n        \nimport pandas as pd\n\ndef test_df_correctly_loaded():\n    correct_df = pd.read_csv(\'datasets/bank_data.csv\')\n    \n    assert correct_df.equals(df), \\\n        "The variable df should contain the data in \'datasets/bank_data.csv\'."\n        \n# def test_head_output():\n#     try:\n#         assert "2030" in first_value.to_string()\n#     except AttributeError:\n#         assert False, \\\n#             "Please use df.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n#     except AssertionError:\n#         assert False, \\\n#             "Hmm, the output of the cell is not what we expected. You should see 2030 in the first five rows of the df DataFrame."')


# ## 2. Graphical exploratory data analysis
# <p>The bank has implemented different recovery strategies at different thresholds (\$1000, \$2000, \$3000 and \$5000) where the greater the Expected Recovery Amount, the more effort the bank puts into contacting the customer. Zeroing in on the first transition (between Level 0 and Level 1) means we are focused on the population with Expected Recovery Amounts between \$0 and \$2000 where the transition between Levels occurred at \$1000. We know that the customers in Level 1 (expected recovery amounts between \$1001 and \$2000) received more attention from the bank and, by definition, they had higher Expected Recovery Amounts than the customers in Level 0 (between \$1 and \$1000).</p>
# <p>Here's a quick summary of the Levels and thresholds again:</p>
# <ul>
# <li>Level 0: Expected recovery amounts &gt;\$0 and &lt;=\$1000</li>
# <li>Level 1: Expected recovery amounts &gt;\$1000 and &lt;=\$2000</li>
# <li>The threshold of \$1000 separates Level 0 from Level 1</li>
# </ul>
# <p>A key question is whether there are other factors besides Expected Recovery Amount that also varied systematically across the \$1000 threshold. For example, does the customer age show a jump (discontinuity) at the \$1000 threshold or does that age vary smoothly? We can examine this by first making a scatter plot of the age as a function of Expected Recovery Amount for a small window of Expected Recovery Amount, \$0 to \$2000. This range covers Levels 0 and 1.</p>

# In[4]:


# Scatter plot of Age vs. Expected Recovery Amount
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x=df['expected_recovery_amount'], y=df['age'], c="g", s=2)
plt.xlim(0, 2000)
plt.ylim(0, 60)
plt.xlabel("Expected Recovery Amount")
plt.ylabel("Age")
plt.legend(loc=2)
plt.show()


# In[5]:


get_ipython().run_cell_magic('nose', '', '\n# no tests for plots\n\n# def test_nothing():\n#     assert True, "Nothing to test."\n\ndef test_matplotlib_loaded_2():\n    assert \'plt\' in globals(), \\\n    \'Did you import the pyplot module from matplotlib under the alias plt?\'')


# ## 3. Statistical test:  age vs. expected recovery amount
# <p>We want to convince ourselves that variables such as age and sex are similar above and below the \$1000 Expected Recovery Amount threshold. This is important because we want to be able to conclude that differences in the actual recovery amount are due to the higher Recovery Strategy and not due to some other difference like age or sex.</p>
# <p>The scatter plot of age versus Expected Recovery Amount did not show an obvious jump around \$1000.  We will now do statistical analysis examining the average age of the customers just above and just below the threshold. We can start by exploring the range from \$900 to \$1100.</p>
# <p>For determining if there is a difference in the ages just above and just below the threshold, we will use the Kruskal-Wallis test, a statistical test that makes no distributional assumptions.</p>

# In[6]:


# Import stats module
from scipy import stats

# Compute average age just below and above the threshold
era_900_1100 = df.loc[(df['expected_recovery_amount']<1100) & 
                      (df['expected_recovery_amount']>=900)]
by_recovery_strategy = era_900_1100.groupby(['recovery_strategy'])
by_recovery_strategy['age'].describe().unstack()

# Perform Kruskal-Wallis test 
Level_0_age = era_900_1100.loc[df['recovery_strategy']=="Level 0 Recovery"]['age']
Level_1_age = era_900_1100.loc[df['recovery_strategy']=="Level 1 Recovery"]['age']
stats.kruskal(Level_0_age, Level_1_age)


# In[7]:


get_ipython().run_cell_magic('nose', '', '\ndef test_stats_loaded():\n    assert \'stats\' in globals(), \\\n    \'Did you import the stats module from scipy?\'\n\ndef test_level_0():\n    correct_Level_0_age_mean= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900) & \n                     (df[\'recovery_strategy\']=="Level 0 Recovery")][\'age\'].mean()\n    Level_0_age_mean= Level_0_age.mean()\n    assert correct_Level_0_age_mean == Level_0_age_mean, \\\n        "The mean age for Level_0_age appears to be incorrect. Did you correctly assign Level_0_age?"\n\ndef test_level_1():\n    correct_Level_1_age_mean= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900) & \n                     (df[\'recovery_strategy\']=="Level 1 Recovery")][\'age\'].mean()\n    Level_1_age_mean= Level_1_age.mean()\n    assert correct_Level_1_age_mean == Level_1_age_mean, \\\n        "The mean age for Level_1_age appears to be incorrect. Did you correctly assign Level_1_age?"')


# ## 4. Statistical test:  sex vs. expected recovery amount
# <p>We have seen that there is no major jump in the average customer age just above and just 
# below the \$1000 threshold by doing a statistical test as well as exploring it graphically with a scatter plot.  </p>
# <p>We want to also test that the percentage of customers that are male does not jump across the \$1000 threshold. We can start by exploring the range of \$900 to \$1100 and later adjust this range.</p>
# <p>We can examine this question statistically by developing cross-tabs as well as doing chi-square tests of the percentage of customers that are male vs. female.</p>

# In[8]:


# Number of customers in each category
crosstab = pd.crosstab(df.loc[(df['expected_recovery_amount']<1100) & 
                              (df['expected_recovery_amount']>=900)]['recovery_strategy'], 
                               df['sex'])
print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)

p_val


# In[9]:


get_ipython().run_cell_magic('nose', '', '\ndef test_crosstab():\n    correct_crosstab = pd.crosstab(df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900)][\'recovery_strategy\'], df[\'sex\'])\n    assert correct_crosstab.equals(crosstab), \\\n    "The crosstab should select the expected_recovery_amount <1100 and >=900."\n\ndef test_pval():\n    chi2_stat, correct_p_val, dof, ex = stats.chi2_contingency(crosstab)\n    assert correct_p_val==p_val, \\\n    "The chi-square test function should use crosstab as the input variable."')


# ## 5. Exploratory graphical analysis: recovery amount
# <p>We are now reasonably confident that customers just above and just below the \$1000 threshold are, on average, similar in their average age and the percentage that are male.  </p>
# <p>It is now time to focus on the key outcome of interest, the actual recovery amount.</p>
# <p>A first step in examining the relationship between the actual recovery amount and the expected recovery amount is to develop a scatter plot where we want to focus our attention at the range just below and just above the threshold. Specifically, we will develop a scatter plot of  Expected Recovery Amount (X) versus Actual Recovery Amount (Y) for Expected Recovery Amounts between \$900 to \$1100.  This range covers Levels 0 and 1.  A key question is whether or not we see a discontinuity (jump) around the \$1000 threshold.</p>

# In[10]:


# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount 
plt.scatter(x=df['expected_recovery_amount'], y=df['actual_recovery_amount'], c="g", s=2)
plt.xlim(900, 1100)
plt.ylim(0, 2000)
plt.xlabel("Expected Recovery Amount")
plt.ylabel("Actual Recovery Amount")
plt.legend(loc=2)
plt.show()


# In[11]:


get_ipython().run_cell_magic('nose', '', '\n# no tests for plots\n\n# def test_nothing():\n#     assert True, "Nothing to test."\n\ndef test_matplotlib_loaded_5():\n    assert \'plt\' in globals(), \\\n    \'Did you import the pyplot module from matplotlib under the alias plt?\'')


# ## 6. Statistical analysis:  recovery amount
# <p>As we did with age, we can perform statistical tests to see if the actual recovery amount has a discontinuity above the \$1000 threshold. We are going to do this for two different windows of the expected recovery amount \$900 to \$1100 and for a narrow range of \$950 to \$1050 to see if our results are consistent.</p>
# <p>Again, we will use the Kruskal-Wallis test.</p>
# <p>We will first compute the average actual recovery amount for those customers just below and just above the threshold using a range from \$900 to \$1100.  Then we will perform a Kruskal-Wallis test to see if the actual recovery amounts are different just above and just below the threshold.  Once we do that, we will repeat these steps for a smaller window of \$950 to \$1050.</p>

# In[12]:


# Compute average actual recovery amount just below and above the threshold
by_recovery_strategy['actual_recovery_amount'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_actual = era_900_1100.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_900_1100.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
print(stats.kruskal(Level_0_actual, Level_1_actual))

# Repeat for a smaller range of $950 to $1050
era_950_1050 = df.loc[(df['expected_recovery_amount']<1050) & 
                      (df['expected_recovery_amount']>=950)]
Level_0_actual = era_950_1050.loc[df['recovery_strategy']=='Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_950_1050.loc[df['recovery_strategy']=='Level 1 Recovery']['actual_recovery_amount']
stats.kruskal(Level_0_actual, Level_1_actual)


# In[13]:


get_ipython().run_cell_magic('nose', '', '\ndef test_level_0():\n    correct_Level_0_actual_mean= df.loc[(df[\'expected_recovery_amount\']<1050) & (df[\'expected_recovery_amount\']>=950) & \n                     (df[\'recovery_strategy\']=="Level 0 Recovery")][\'actual_recovery_amount\'].mean()\n    Level_0_actual_mean= Level_0_actual.mean()\n    assert correct_Level_0_actual_mean == Level_0_actual_mean, \\\n        "The mean actual_recovery_amount for Level_0_actual appears to be incorrect. Did you correctly assign Level_0_actual?"\n\ndef test_level_1():\n    correct_Level_1_actual_mean= df.loc[(df[\'expected_recovery_amount\']<1050) & (df[\'expected_recovery_amount\']>=950) & \n                     (df[\'recovery_strategy\']=="Level 1 Recovery")][\'actual_recovery_amount\'].mean()\n    Level_1_actual_mean= Level_1_actual.mean()\n    assert correct_Level_1_actual_mean == Level_1_actual_mean, \\\n        "The mean actual_recovery_amount for Level_1_actual appears to be incorrect. Did you correctly assign Level_1_actual?"')


# ## 7. Regression modeling: no threshold
# <p>We now want to take a regression-based approach to estimate the program impact at the \$1000 threshold using data that is just above and below the threshold. </p>
# <p>We will build two models. The first model does not have a threshold while the second will include a threshold.</p>
# <p>The first model predicts the actual recovery amount (dependent variable) as a function of the expected recovery amount (independent variable). We expect that there will be a strong positive relationship between these two variables.  </p>
# <p>We will examine the adjusted R-squared to see the percent of variance explained by the model.  In this model, we are not representing the threshold but simply seeing how the variable used for assigning the customers (expected recovery amount) relates to the outcome variable (actual recovery amount).</p>

# In[14]:


# Import statsmodels
import statsmodels.api as sm

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the model summary statistics
model.summary()


# In[15]:


get_ipython().run_cell_magic('nose', '', '\ndef test_x():\n    correct_x= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900)][\'expected_recovery_amount\']\n    correct_x = sm.add_constant(correct_x)\n    assert correct_x[\'expected_recovery_amount\'].mean() == X[\'expected_recovery_amount\'].mean(), \\\n        "The mean expected_recovery_amount for X appears incorrect. Check your assignment of X.  It should include expected_recovery_amount and indicator_1000 when the expected_recovery_amount is <1100 and >=900."\ndef test_y():\n    correct_y= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900)][\'actual_recovery_amount\']\n    assert correct_y.mean() == y.mean(), \\\n        "The mean actual_recovery_amount for y appears incorrect. Check your assignment of y. It should include the actual_recovery_amount when the expected_recovery_amount is <1100 and >=900."\n        \n# def test_df_correctly_loaded():\n#     correct_model = sm.OLS(y,x).fit()\n#     assert correct_model.params[1] == model.params[1], \\\n#         "Check your assignment of model. It should be equal to sm.OLS(y,X).fit()."')


# ## 8. Regression modeling: adding true threshold
# <p>From the first model, we see that the expected recovery amount's regression coefficient is statistically significant. </p>
# <p>The second model adds an indicator of the true threshold to the model (in this case at \$1000).  </p>
# <p>We will create an indicator variable (either a 0 or a 1) that represents whether or not the expected recovery amount was greater than \$1000. When we add the true threshold to the model, the regression coefficient for the true threshold represents the additional amount recovered due to the higher recovery strategy.  That is to say, the regression coefficient for the true threshold measures the size of the discontinuity for customers just above and just below the threshold.</p>
# <p>If the higher recovery strategy helped recovery more money, then the regression coefficient of the true threshold will be greater than zero.  If the higher recovery strategy did not help recovery more money, then the regression coefficient will not be statistically significant.</p>

# In[16]:


# Create indicator (0 or 1) for expected recovery amount >= $1000
df['indicator_1000'] = np.where(df['expected_recovery_amount']<1000, 0, 1)
era_900_1100 = df.loc[(df['expected_recovery_amount']<1100) & 
                      (df['expected_recovery_amount']>=900)]

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y,X).fit()

# Print the model summary
model.summary()


# In[17]:


get_ipython().run_cell_magic('nose', '', '\ndef test_x():\n    correct_x= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900),\n           [\'expected_recovery_amount\',\'indicator_1000\']]\n    correct_x = sm.add_constant(correct_x)\n    assert correct_x[\'expected_recovery_amount\'].mean() == X[\'expected_recovery_amount\'].mean(), \\\n        "The mean expected_recovery_amount for X appears incorrect. Check your assignment of X.  It should include expected_recovery_amount and indicator_1000 when the expected_recovery_amount is <1100 and >=900."\ndef test_y():\n    correct_y= df.loc[(df[\'expected_recovery_amount\']<1100) & (df[\'expected_recovery_amount\']>=900)][\'actual_recovery_amount\']\n    assert correct_y.mean() == y.mean(), \\\n        "The mean actual_recovery_amount for y appears incorrect. Check your assignment of y. It should include the actual_recovery_amount when the expected_recovery_amount is <1100 and >=900."\n        \n# def test_df_correctly_loaded():\n#     correct_model = sm.OLS(y,X).fit()\n#     assert correct_model.params[1] == model.params[1], \\\n#         "Check your assignment of model. It should be equal to sm.OLS(y,X).fit()."')


# ## 9. Regression modeling: adjusting the window
# <p>The regression coefficient for the true threshold was statistically significant with an estimated impact of around \$278.  This is much larger than the \$50 per customer needed to run this higher recovery strategy. </p>
# <p>Before showing this to our manager, we want to convince ourselves that this result wasn't due to choosing an expected recovery amount window of \$900 to \$1100. Let's repeat this analysis for the window from \$950 to \$1050 to see if we get similar results.</p>
# <p>The answer? Whether we use a wide (\$900 to \$1100) or narrower window (\$950 to \$1050), the incremental recovery amount at the higher recovery strategy is much greater than the \$50 per customer it costs for the higher recovery strategy.  So we conclude that the higher recovery strategy is worth the extra cost of \$50 per customer.</p>

# In[18]:


# Redefine era_950_1050 so the indicator variable is included
era_950_1050 = df.loc[(df['expected_recovery_amount']<1050) & 
                      (df['expected_recovery_amount']>=950)]

# Define X and y 
X = era_950_1050[['expected_recovery_amount','indicator_1000']]
y = era_950_1050['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y,X).fit()

# Print the model summary
model.summary()


# In[19]:


get_ipython().run_cell_magic('nose', '', '\ndef test_x():\n    correct_x= df.loc[(df[\'expected_recovery_amount\']<1050) & (df[\'expected_recovery_amount\']>=950),[\'expected_recovery_amount\',\'indicator_1000\']]\n    correct_x = sm.add_constant(correct_x)\n    assert correct_x[\'expected_recovery_amount\'].mean() == X[\'expected_recovery_amount\'].mean(), \\\n        "The mean expected_recovery_amount for X appears incorrect. Check your assignment of X.  It should include expected_recovery_amount and indicator_1000 when the expected_recovery_amount is <1050 and >=950."\ndef test_y():\n    correct_y= df.loc[(df[\'expected_recovery_amount\']<1050) & (df[\'expected_recovery_amount\']>=950)][\'actual_recovery_amount\']\n    assert correct_y.mean() == y.mean(), \\\n        "The mean actual_recovery_amount for y appears incorrect. Check your assignment of y. It should include the actual_recovery_amount when the expected_recovery_amount is <1050 and >=950."\n        \n# def test_df_correctly_loaded():\n#     correct_model = sm.OLS(y,X).fit()\n#     assert correct_model.params[1] == model.params[1], \\\n#         "Check your assignment of model.  It should be equal to sm.OLS(y,X).fit()."')

