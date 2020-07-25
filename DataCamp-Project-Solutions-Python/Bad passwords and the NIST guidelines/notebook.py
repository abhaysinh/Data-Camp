
# coding: utf-8

# ## 1. The NIST Special Publication 800-63B
# <p>If you – 50 years ago – needed to come up with a secret password you were probably part of a secret espionage organization or (more likely) you were pretending to be a spy when playing as a kid. Today, many of us are forced to come up with new passwords <em>all the time</em> when signing into sites and apps. As a password <em>inventeur</em> it is your responsibility to come up with good, hard-to-crack passwords. But it is also in the interest of sites and apps to make sure that you use good passwords. The problem is that it's really hard to define what makes a good password. However, <em>the National Institute of Standards and Technology</em> (NIST) knows what the second best thing is: To make sure you're at least not using a <em>bad</em> password. </p>
# <p>In this notebook, we will go through the rules in <a href="https://pages.nist.gov/800-63-3/sp800-63b.html">NIST Special Publication 800-63B</a> which details what checks a <em>verifier</em> (what the NIST calls a second party responsible for storing and verifying passwords) should perform to make sure users don't pick bad passwords. We will go through the passwords of users from a fictional company and use python to flag the users with bad passwords. But us being able to do this already means the fictional company is breaking one of the rules of 800-63B:</p>
# <blockquote>
#   <p>Verifiers SHALL store memorized secrets in a form that is resistant to offline attacks. Memorized secrets SHALL be salted and hashed using a suitable one-way key derivation function.</p>
# </blockquote>
# <p>That is, never save users' passwords in plaintext, always encrypt the passwords! Keeping this in mind for the next time we're building a password management system, let's load in the data.</p>
# <p><em>Warning: The list of passwords and the fictional user database both contain <strong>real</strong> passwords leaked from <strong>real</strong> websites. These passwords have not been filtered in any way and include words that are explicit, derogatory and offensive.</em></p>

# In[2]:


# Importing the pandas module
import pandas as pd

# Loading in datasets/users.csv 
users = pd.read_csv('datasets/users.csv')

# Printing out how many users we've got
users = pd.read_csv('datasets/users.csv')

# Taking a look at the 12 first users
users.head(12)


# In[3]:


get_ipython().run_cell_magic('nose', '', '\nimport pandas as pd\n\ndef test_users_read_in_correctly():\n    correct_users = pd.read_csv("datasets/users.csv")\n    assert correct_users.equals(users), \\\n    \'users should contain the data in "datasets/users.csv"\'')


# ## 2. Passwords should not be too short
# <p>If we take a look at the first 12 users above we already see some bad passwords. But let's not get ahead of ourselves and start flagging passwords <em>manually</em>. What is the first thing we should check according to the NIST Special Publication 800-63B?</p>
# <blockquote>
#   <p>Verifiers SHALL require subscriber-chosen memorized secrets to be at least 8 characters in length.</p>
# </blockquote>
# <p>Ok, so the passwords of our users shouldn't be too short. Let's start by checking that!</p>

# In[4]:


# Calculating the lengths of users' passwords
users['length'] = users['password'].str.len()

# Flagging the users with too short passwords
users['too_short'] = users['length']<8

# Counting and printing the number of users with too short passwords
print(sum(users['too_short']))

# Taking a look at the 12 first rows
users.head(12)


# In[5]:


get_ipython().run_cell_magic('nose', '', '\ndef test_length_sum_correct():\n    assert (users[\'password\'].str.len() < 8).sum() == users[\'too_short\'].sum(), \\\n    "users[\'too_short\'] should be a True/False column where all rows with passwords < 8 are True."')


# ## 3.  Common passwords people use
# <p>Already this simple rule flagged a couple of offenders among the first 12 users. Next up in Special Publication 800-63B is the rule that</p>
# <blockquote>
#   <p>verifiers SHALL compare the prospective secrets against a list that contains values known to be commonly-used, expected, or compromised.</p>
#   <ul>
#   <li>Passwords obtained from previous breach corpuses.</li>
#   <li>Dictionary words.</li>
#   <li>Repetitive or sequential characters (e.g. ‘aaaaaa’, ‘1234abcd’).</li>
#   <li>Context-specific words, such as the name of the service, the username, and derivatives thereof.</li>
#   </ul>
# </blockquote>
# <p>We're going to check these in order and start with <em>Passwords obtained from previous breach corpuses</em>, that is, websites where hackers have leaked all the users' passwords. As many websites don't follow the NIST guidelines and encrypt passwords there now exist large lists of the most popular passwords. Let's start by loading in the 10,000 most common passwords which I've taken from <a href="https://github.com/danielmiessler/SecLists/tree/master/Passwords">here</a>.</p>

# In[6]:


# Reading in the top 10000 passwords
common_passwords = pd.read_csv('datasets/10_million_password_list_top_10000.txt', header=None, squeeze=True)


# Taking a look at the top 20
common_passwords.head(20)


# In[7]:


get_ipython().run_cell_magic('nose', '', '\ndef test_common_passwords_correct():\n    correct_common_passwords = pd.read_csv("datasets/10_million_password_list_top_10000.txt",\n                                           header=None, squeeze=True)\n    assert correct_common_passwords.equals(common_passwords), \\\n    \'datasets/10_million_password_list_top_10000.txt should be read as a Series and put into common_passwords.\'')


# ## 4.  Passwords should not be common passwords
# <p>The list of passwords was ordered, with the most common passwords first, and so we shouldn't be surprised to see passwords like <code>123456</code> and <code>qwerty</code> above. As hackers also have access to this list of common passwords, it's important that none of our users use these passwords!</p>
# <p>Let's flag all the passwords in our user database that are among the top 10,000 used passwords.</p>

# In[8]:


# Flagging the users with passwords that are common passwords
users['common_password'] = users['password'].isin(common_passwords)

# Counting and printing the number of users using common passwords
print(sum(users['common_password']))

# Taking a look at the 12 first rows
users.head(12)


# In[9]:


get_ipython().run_cell_magic('nose', '', '\ndef test_example():\n    assert users[\'password\'].isin(common_passwords).sum() == users[\'common_password\'].sum(), \\\n    "users[\'common_password\'] should be True for each row with a password that is also in common_passwords."')


# ## 5. Passwords should not be common words
# <p>Ay ay ay! It turns out many of our users use common passwords, and of the first 12 users there are already two. However, as most common passwords also tend to be short, they were already flagged as being too short. What is the next thing we should check?</p>
# <blockquote>
#   <p>Verifiers SHALL compare the prospective secrets against a list that contains [...] dictionary words.</p>
# </blockquote>
# <p>This follows the same logic as before: It is easy for hackers to check users' passwords against common English words and therefore common English words make bad passwords. Let's check our users' passwords against the top 10,000 English words from <a href="https://github.com/first20hours/google-10000-english">Google's Trillion Word Corpus</a>.</p>

# In[10]:


# Reading in a list of the 10000 most common words
words = pd.read_csv('datasets/google-10000-english.txt', header=None, squeeze=True)

# Flagging the users with passwords that are common words
users['common_word'] = users['password'].str.lower().isin(words)

# Counting and printing the number of users using common words as passwords
print(sum(users['common_word']))

# Taking a look at the 12 first rows
users.head(12)


# In[11]:


get_ipython().run_cell_magic('nose', '', '\ndef test_words_correct():\n    correct_words = pd.read_csv("datasets/google-10000-english.txt",\n                    header=None, squeeze=True)\n    assert correct_words.equals(words), \\\n    \'datasets/google-10000-english.txt should be read in as a Series and put into words.\'\n    \ndef test_common_words_correct():\n    assert users[\'password\'].str.lower().isin(words).sum() == users[\'common_word\'].sum() , \\\n    "users[\'common_word\'] should be True for each row with a password that is also in words."')


# ## 6. Passwords should not be your name
# <p>It turns out many of our passwords were common English words too! Next up on the NIST list:</p>
# <blockquote>
#   <p>Verifiers SHALL compare the prospective secrets against a list that contains [...] context-specific words, such as the name of the service, the username, and derivatives thereof.</p>
# </blockquote>
# <p>Ok, so there are many things we could check here. One thing to notice is that our users' usernames consist of their first names and last names separated by a dot. For now, let's just flag passwords that are the same as either a user's first or last name.</p>

# In[12]:


# Extracting first and last names into their own columns
users['first_name'] = users['user_name'].str.extract(r'(^\w+)', expand=False)
users['last_name'] = users['user_name'].str.extract(r'(\w+$)', expand=False)

# Flagging the users with passwords that matches their names
users['uses_name'] = (users['first_name'].str.lower() == users['password']) | ((users['last_name']).str.lower()== users['password'])

# Counting and printing the number of users using names as passwords
# ... YOUR CODE FOR TASK 6 ...
print(sum(users['uses_name']))

# Taking a look at the 12 first rows
# ... YOUR CODE FOR TASK 6 ...
users[['user_name','first_name', 'last_name','password','uses_name']]


# In[13]:


get_ipython().run_cell_magic('nose', '', '\ndef test_not_same_as_name():\n    correct_first_name = users[\'user_name\'].str.extract(r\'(^\\w+)\', expand = False)\n    correct_last_name = users[\'user_name\'].str.extract(r\'(\\w+$)\', expand = False)\n\n    # Flagging the users with passwords that matches their names\n    correct_uses_name = (\n        (users[\'password\'].str.lower() == users[\'first_name\']) |\n        (users[\'password\'].str.lower() == users[\'last_name\']))\n    \n    assert correct_uses_name.sum() == users[\'uses_name\'].sum(), \\\n    "users[\'uses_name\'] should be True for each row with a password which is also the first or last name."')


# ## 7. Passwords should not be repetitive
# <p>Milford Hubbard (user number 12 above), what where you thinking!? Ok, so the last thing we are going to check is a bit tricky:</p>
# <blockquote>
#   <p>verifiers SHALL compare the prospective secrets [so that they don't contain] repetitive or sequential characters (e.g. ‘aaaaaa’, ‘1234abcd’).</p>
# </blockquote>
# <p>This is tricky to check because what is <em>repetitive</em> is hard to define. Is <code>11111</code> repetitive? Yes! Is <code>12345</code> repetitive? Well, kind of. Is <code>13579</code> repetitive? Maybe not..? To check for <em>repetitiveness</em> can be arbitrarily complex, but here we're only going to do something simple. We're going to flag all passwords that contain 4 or more repeated characters.</p>

# In[14]:


### Flagging the users with passwords with >= 4 repeats
users['too_many_repeats'] = users['password'].str.contains(r'(.)\1\1\1')

# Taking a look at the users with too many repeats
users[users['too_many_repeats']==True]


# In[15]:


get_ipython().run_cell_magic('nose', '', '\ndef test_too_many_repeats():\n    assert users[\'password\'].str.contains(r\'(.)\\1\\1\\1\').sum() == users[\'too_many_repeats\'].sum(), \\\n    "users[\'too_many_repeats\'] should be True for each row with a password with 4 or more repeats."')


# ## 8. All together now!
# <p>Now we have implemented all the basic tests for bad passwords suggested by NIST Special Publication 800-63B! What's left is just to flag all bad passwords and maybe to send these users an e-mail that strongly suggests they change their password.</p>

# In[16]:


# Flagging all passwords that are bad
users['bad_password'] = (users['too_short'])|(users['common_password'])|(users['common_word'])|(users['uses_name'])|(users['too_many_repeats'])


# Counting and printing the number of bad passwords
print(sum(users['bad_password']))

# Looking at the first 25 bad passwords
users[users['bad_password']==True]['password'].head(25)


# In[17]:


get_ipython().run_cell_magic('nose', '', '\ndef test_all_nist_rules():\n    correct_bad_password = ( \n        users[\'too_short\'] | \n        users[\'common_password\'] |\n        users[\'common_word\'] |\n        users[\'uses_name\'] |\n        users[\'too_many_repeats\'] )\n    assert correct_bad_password.sum() == users[\'bad_password\'].sum(), \\\n    "All rows with passwords that should be flagged as bad should have users[\'bad_password\'] set to True."')


# ## 9. Otherwise, the password should be up to the user
# <p>In this notebook, we've implemented the password checks recommended by the NIST Special Publication 800-63B. It's certainly possible to better implement these checks, for example, by using a longer list of common passwords. Also note that the NIST checks in no way guarantee that a chosen password is good, just that it's not obviously bad.</p>
# <p>Apart from the checks we've implemented above the NIST is also clear with what password rules should <em>not</em> be imposed:</p>
# <blockquote>
#   <p>Verifiers SHOULD NOT impose other composition rules (e.g., requiring mixtures of different character types or prohibiting consecutively repeated characters) for memorized secrets. Verifiers SHOULD NOT require memorized secrets to be changed arbitrarily (e.g., periodically).</p>
# </blockquote>
# <p>So the next time a website or app tells you to "include both a number, symbol and an upper and lower case character in your password" you should send them a copy of <a href="https://pages.nist.gov/800-63-3/sp800-63b.html">NIST Special Publication 800-63B</a>.</p>

# In[18]:


# Enter a password that passes the NIST requirements
# PLEASE DO NOT USE AN EXISTING PASSWORD HERE
new_password = "test@1986"


# In[19]:


get_ipython().run_cell_magic('nose', '', '\ndef test_not_bad_password():\n    temp_new_password = pd.Series(new_password)\n    temp_common_passwords = pd.read_csv("datasets/10_million_password_list_top_10000.txt",\n                                   header=None, squeeze=True)\n    temp_words = pd.read_csv("datasets/google-10000-english.txt",\n                        header=None, squeeze=True)\n\n    is_bad = (\n        (temp_new_password.str.len() < 8) |\n        (temp_new_password.isin(temp_common_passwords)) |\n        (temp_new_password.str.lower().isin(temp_words)) |\n        (temp_new_password.str.contains(r\'(.)\\1\\1\\1\'))\n    ).all()\n    assert not is_bad, \\\n    \'This password does not fulfill the NIST requirements.\'')

