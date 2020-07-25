
# coding: utf-8

# ## 1. Tweet classification: Trump vs. Trudeau
# <p>So you think you can classify text? How about tweets? In this notebook, we'll take a dive into the world of social media text classification by investigating how to properly classify tweets from two prominent North American politicians: Donald Trump and Justin Trudeau.</p>
# <p><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/President_Donald_Trump_and_Prime_Minister_Justin_Trudeau_Joint_Press_Conference%2C_February_13%2C_2017.jpg/800px-President_Donald_Trump_and_Prime_Minister_Justin_Trudeau_Joint_Press_Conference%2C_February_13%2C_2017.jpg" alt="Donald Trump and Justin Trudeau shaking hands." height="50%" width="50%"></p>
# <p><a href="https://commons.wikimedia.org/wiki/File:President_Donald_Trump_and_Prime_Minister_Justin_Trudeau_Joint_Press_Conference,_February_13,_2017.jpg">Photo Credit: Executive Office of the President of the United States</a></p>
# <p>Tweets pose specific problems to NLP, including the fact they are shorter texts. There are also plenty of platform-specific conventions to give you hassles: mentions, #hashtags, emoji, links and short-hand phrases (ikr?). Can we overcome those challenges and build a useful classifier for these two tweeters? Yes! Let's get started.</p>
# <p>To begin, we will import all the tools we need from scikit-learn. We will need to properly vectorize our data (<code>CountVectorizer</code> and <code>TfidfVectorizer</code>). And we will also want to import some models, including <code>MultinomialNB</code> from the <code>naive_bayes</code> module, <code>LinearSVC</code> from the <code>svm</code> module and <code>PassiveAggressiveClassifier</code> from the <code>linear_model</code> module. Finally, we'll need <code>sklearn.metrics</code> and <code>train_test_split</code> and <code>GridSearchCV</code> from the <code>model_selection</code> module to evaluate and optimize our model.</p>

# In[2]:


# Set seed for reproducibility
import random; random.seed(53)

# Import all we need from sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics


# In[3]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code.\n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_CountVectorizer():\n    assert 'CountVectorizer' in globals(), \\\n    'CountVectorizer should be imported.'\n    \ndef test_TfidfVectorizer():\n    assert 'TfidfVectorizer' in globals(), \\\n    'TfidfVectorizer should be imported.'\n    \ndef test_train_test_split():\n    assert 'train_test_split' in globals(), \\\n    'train_test_split should be imported.'\n    \ndef test_MultinomialNB():\n    assert 'MultinomialNB' in globals(), \\\n    'MultinomialNB should be imported.'\n\ndef test_LinearSVC():\n    assert 'LinearSVC' in globals(), \\\n    'LinearSVC should be imported.'\n\ndef test_metrics():\n    assert 'metrics' in globals(), \\\n    'metrics should be imported.'")


# ## 2. Transforming our collected data
# <p>To begin, let's start with a corpus of tweets which were collected in November 2017. They are available in CSV format. We'll use a Pandas DataFrame to help import the data and pass it to scikit-learn for further processing.</p>
# <p>Since the data has been collected via the Twitter API and not split into test and training sets, we'll need to do this. Let's use <code>train_test_split()</code> with <code>random_state=53</code> and a test size of 0.33, just as we did in the DataCamp course. This will ensure we have enough test data and we'll get the same results no matter where or when we run this code.</p>

# In[4]:


import pandas as pd

# Load data
tweet_df = pd.read_csv('datasets/tweets.csv')

# Create target
y = tweet_df['author']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(tweet_df['status'], y, random_state=53, test_size=.33)


# In[5]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_df():\n    assert isinstance(tweet_df, pd.DataFrame), \\\n    'tweet_df should be a Pandas DataFrame.'\n\ndef test_y():\n    assert isinstance(y, pd.Series), \\\n    'y should be a Pandas Series.'\n\ndef train_test_split_test():\n    assert len(y_train) == len(X_train), \\\n    'Make sure to run the train-test split.'\n    assert len(y_test) == len(X_test), \\\n    'Make sure to run the train-test split.'")


# ## 3. Vectorize the tweets
# <p>We have the training and testing data all set up, but we need to create vectorized representations of the tweets in order to apply machine learning.</p>
# <p>To do so, we will utilize the <code>CountVectorizer</code> and <code>TfidfVectorizer</code> classes which we will first need to fit to the data.</p>
# <p>Once this is complete, we can start modeling with the new vectorized tweets!</p>

# In[6]:


# Initialize count vectorizer
count_vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=0.05)

# Create count train and test variables
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Initialize tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=0.05)

# Create tfidf train and test variables
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)


# In[7]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nimport scipy\n\ndef test_train():\n    assert isinstance(count_train, scipy.sparse.csr.csr_matrix), \\\n    'Make sure to run the count vectorizer for the training data.'\n    assert isinstance(tfidf_train, scipy.sparse.csr.csr_matrix), \\\n    'Make sure to run the TFIDF vectorizer for the training data.'\n\ndef test_test():\n    assert isinstance(count_test, scipy.sparse.csr.csr_matrix), \\\n    'Make sure to run the count vectorizer for the test data.'\n    assert isinstance(tfidf_test, scipy.sparse.csr.csr_matrix), \\\n    'Make sure to run the TFIDF vectorizer for the test data.'\n    \ndef test_vectorizers():\n    assert isinstance(tfidf_vectorizer, TfidfVectorizer), \\\n    'tfidf_vectorizer is missing or an incorrect type.'\n    assert isinstance(count_vectorizer, CountVectorizer), \\\n    'count_vectorizer is missing or an incorrect type.'\n    assert tfidf_vectorizer.stop_words == 'english', \\\n    'Use parameters to set the stop words for the TFIDF vectorizer.'\n    assert count_vectorizer.stop_words == 'english', \\\n    'Use parameters to set the stop words for the count vectorizer.'\n    assert tfidf_vectorizer.max_df == 0.9, \\\n    'Use parameters to set the max_df for the TFIDF vectorizer.'\n    assert count_vectorizer.max_df == 0.9, \\\n    'Use parameters to set the max_df for the count vectorizer.'\n    assert tfidf_vectorizer.min_df == 0.05, \\\n    'Use parameters to set the min_df for the TFIDF vectorizer.'\n    assert count_vectorizer.min_df == 0.05, \\\n    'Use parameters to set the min_df for the count vectorizer.'")


# ## 4. Training a multinomial naive Bayes model
# <p>Now that we have the data in vectorized form, we can train the first model. Investigate using the Multinomial Naive Bayes model with both the <code>CountVectorizer</code> and <code>TfidfVectorizer</code> data. Which do will perform better? How come?</p>
# <p>To assess the accuracies, we will print the test sets accuracy scores for both models.</p>

# In[8]:


# Create a MulitnomialNB model
tfidf_nb = MultinomialNB()
tfidf_nb.fit(tfidf_train, y_train)

# Run predict on your TF-IDF test data to get your predictions
tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

# Calculate the accuracy of your predictions
tfidf_nb_score = metrics.accuracy_score(tfidf_nb_pred, y_test)

# Create a MulitnomialNB model
count_nb = MultinomialNB()
count_nb.fit(count_train, y_train)

# Run predict on your count test data to get your predictions
count_nb_pred = count_nb.predict(count_test)

# Calculate the accuracy of your predictions
count_nb_score = metrics.accuracy_score(count_nb_pred, y_test)

print('NaiveBayes Tfidf Score: ', tfidf_nb_score)
print('NaiveBayes Count Score: ', count_nb_score)


# In[9]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nimport numpy\n\ndef test_models():\n    assert isinstance(count_nb, MultinomialNB), \\\n    'count_nb should be a MultinomialNB model.'\n    assert isinstance(tfidf_nb, MultinomialNB), \\\n    'tfidf_nb should be a MultinomialNB model.'\n    assert isinstance(count_nb.classes_, numpy.ndarray)\n    assert len(count_nb.classes_) == 2, \\\n    'count_nb should have only two classes.'\n    assert isinstance(tfidf_nb.classes_, numpy.ndarray)\n    assert len(tfidf_nb.classes_) == 2, \\\n    'tfidf_nb should have only two classes.' \n    \n\ndef test_pred():\n    assert isinstance(tfidf_nb_pred, numpy.ndarray), \\\n    'tfidf_nb_pred should be a numpy array.'\n    assert isinstance(count_nb_pred, numpy.ndarray), \\\n    'count_nb_pred should be a numpy array.'\n    assert set(tfidf_nb_pred) == set(tfidf_nb.classes_), \\\n    'tfidf_nb_pred should use the same classes as the model for prediction.'\n    assert set(count_nb_pred) == set(count_nb.classes_), \\\n    'count_nb_pred should use the same classes as the model for prediction.'\n\ndef test_score():\n    assert isinstance(tfidf_nb_score, float), \\\n    'tfidf_nb_score should be a float.'\n    assert isinstance(count_nb_score, float), \\\n    'count_nb_score should be a float.'\n    assert tfidf_nb_score > .802, \\\n    'tfidf_nb_score should be above .802'\n    assert count_nb_score > .794, \\\n    'count_nb_score should be above .794'")


# ## 5. Evaluating our model using a confusion matrix
# <p>We see that the TF-IDF model performs better than the count-based approach. Based on what we know from the NLP fundamentals course, why might that be? We know that TF-IDF allows unique tokens to have a greater weight - perhaps tweeters are using specific important words that identify them! Let's continue the investigation.</p>
# <p>For classification tasks, an accuracy score doesn't tell the whole picture. A better evaluation can be made if we look at the confusion matrix, which shows the number correct and incorrect classifications based on each class. We can use the metrics, True Positives, False Positives, False Negatives, and True Negatives, to determine how well the model performed on a given class. How many times was Trump misclassified as Trudeau?</p>

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

from datasets.helper_functions import plot_confusion_matrix

# Calculate the confusion matrices for the tfidf_nb model and count_nb models
tfidf_nb_cm = metrics.confusion_matrix(y_test, tfidf_nb_pred, labels=['Donald J. Trump', 'Justin Trudeau'])
count_nb_cm = metrics.confusion_matrix(y_test, count_nb_pred, labels=['Donald J. Trump', 'Justin Trudeau'])

# Plot the tfidf_nb_cm confusion matrix
plot_confusion_matrix(tfidf_nb_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="TF-IDF NB Confusion Matrix")

# Plot the count_nb_cm confusion matrix without overwriting the first plot 
plot_confusion_matrix(count_nb_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="Count NB Confusion Matrix", figure=1)


# In[11]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nimport numpy\n\n\ndef test_cm():\n    assert isinstance(tfidf_nb_cm, numpy.ndarray), \\\n    'tfidf_nb_cm should be a NumPy array.'\n    assert isinstance(count_nb_cm, numpy.ndarray), \\\n    'count_nb_cm should be a NumPy array.'\n    assert tfidf_nb_cm[0][0] == 56, \\\n    'The true label and predicted label for Trump in the TFIDF MultinomialNB model should be 56.'")


# ## 6. Trying out another classifier: Linear SVC
# <p>So the Bayesian model only has one prediction difference between the TF-IDF and count vectorizers -- fairly impressive! Interestingly, there is some confusion when the predicted label is Trump but the actual tweeter is Trudeau. If we were going to use this model, we would want to investigate what tokens are causing the confusion in order to improve the model. </p>
# <p>Now that we've seen what the Bayesian model can do, how about trying a different approach? <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">LinearSVC</a> is another popular choice for text classification. Let's see if using it with the TF-IDF vectors improves the accuracy of the classifier!</p>

# In[12]:


# Create a LinearSVM model
tfidf_svc = LinearSVC()
tfidf_svc.fit(tfidf_train, y_train)

# Run predict on your tfidf test data to get your predictions
tfidf_svc_pred = tfidf_svc.predict(tfidf_test)

# Calculate your accuracy using the metrics module
tfidf_svc_score = metrics.accuracy_score(tfidf_svc_pred, y_test)

print("LinearSVC Score:   %0.3f" % tfidf_svc_score)

# Calculate the confusion matrices for the tfidf_svc model
svc_cm = metrics.confusion_matrix(y_test, tfidf_svc_pred, labels=['Donald J. Trump', 'Justin Trudeau'])

# Plot the confusion matrix using the plot_confusion_matrix function
plot_confusion_matrix(svc_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="TF-IDF LinearSVC Confusion Matrix")


# In[13]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nimport numpy\n\ndef test_models():\n    assert isinstance(tfidf_svc, LinearSVC), \\\n    'tfidf_svc should be a LinearSVC model.'\n    assert isinstance(tfidf_svc.classes_, numpy.ndarray), \\\n    'tfidf_svc should have the proper classes.'\n    assert len(tfidf_svc.classes_) == 2, \\\n    'tfidf_svc should have exactly 2 classes.' \n\ndef test_pred():\n    assert isinstance(tfidf_svc_pred, numpy.ndarray), \\\n    'tfidf_svc_pred should be a numpy array.'\n    assert set(tfidf_svc_pred) == set(tfidf_svc.classes_), \\\n    'tfidf_svc_pred should have the same classes as the model.'\n    \ndef test_score():\n    assert isinstance(tfidf_svc_score, float), \\\n    'tfidf_svc_score should be a float.'\n    assert tfidf_svc_score > .84, \\\n    'tfidf_svc_score should be > .84.' ")


# ## 7. Introspecting our top model
# <p>Wow, the LinearSVC model is even better than the Multinomial Bayesian one. Nice work! Via the confusion matrix we can see that, although there is still some confusion where Trudeau's tweets are classified as Trump's, the False Positive rate is better than the previous model. So, we have a performant model, right? </p>
# <p>We might be able to continue tweaking and improving all of the previous models by learning more about parameter optimization or applying some better preprocessing of the tweets. </p>
# <p>Now let's see what the model has learned. Using the LinearSVC Classifier with two classes (Trump and Trudeau) we can sort the features (tokens), by their weight and see the most important tokens for both Trump and Trudeau. What are the most Trump-like or Trudeau-like words? Did the model learn something useful to distinguish between these two men? </p>

# In[14]:


from datasets.helper_functions import plot_and_return_top_features

# Import pprint from pprint
from pprint import pprint

# Get the top features using the plot_and_return_top_features function and your top model and tfidf vectorizer
top_features = plot_and_return_top_features(tfidf_svc, tfidf_vectorizer)

# pprint the top features
pprint(top_features)


# In[15]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_example():\n    assert isinstance(top_features, list), \\\n    'top_features should be a Python list.'\n    assert isinstance(top_features[0], tuple), \\\n    'The top_features should be a list of tuples.'\n    assert isinstance(top_features[0][0], float), \\\n    'The first element of each tuple in the top_features list should be a float.'\n    assert isinstance(top_features[0][1], str), \\\n    'The second element of each tuple in the top_features list should be a string.'\n    assert top_features[0][1] == 'great', \\\n    'The top feature for Trump (i.e. first feature returned) should be the word: great.'")


# ## 8. Bonus: can you write a Trump or Trudeau tweet?
# <p>So, what did our model learn? It seems like it learned that Trudeau tweets in French!</p>
# <p>I challenge you to write your own tweet using the knowledge gained to trick the model! Use the printed list or plot above to make some inferences about what words will classify your text as Trump or Trudeau. Can you fool the model into thinking you are Trump or Trudeau?</p>
# <p>If you can write French, feel free to make your Trudeau-impersonation tweet in French! As you may have noticed, these French words are common words, or, "stop words". You could remove both English and French stop words from the tweets as a preprocessing step, but that might decrease the accuracy of the model because Trudeau is the only French-speaker in the group. If you had a dataset with more than one French speaker, this would be a useful preprocessing step.</p>
# <p>Future work on this dataset could involve:</p>
# <ul>
# <li>Add extra preprocessing (such as removing URLs or French stop words) and see the effects</li>
# <li>Use GridSearchCV to improve both your Bayesian and LinearSVC models by finding the optimal parameters</li>
# <li>Introspect your Bayesian model to determine what words are more Trump- or Trudeau- like</li>
# <li>Add more recent tweets to your dataset using tweepy and retrain</li>
# </ul>
# <p>Good luck writing your impersonation tweets -- feel free to share them on Twitter!</p>

# In[16]:


# Write two tweets as strings, one which you want to classify as Trump and one as Trudeau
trump_tweet = 'fake news'
trudeau_tweet = 'canada'

# Vectorize each tweet using the TF-IDF vectorizer's transform method
trump_tweet_vectorized = tfidf_vectorizer.transform([trump_tweet])
trudeau_tweet_vectorized = tfidf_vectorizer.transform([trudeau_tweet])

# Call the predict method on your vectorized tweets
trump_tweet_pred = tfidf_svc.predict(trump_tweet_vectorized)
trudeau_tweet_pred = tfidf_svc.predict(trudeau_tweet_vectorized)

print("Predicted Trump tweet", trump_tweet_pred)
print("Predicted Trudeau tweet", trudeau_tweet_pred)


# In[17]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nimport scipy\n\ndef test_example():\n    assert isinstance(trump_tweet, str), \\\n    "trump_tweet should be a Python string"\n    assert isinstance(trudeau_tweet, str), \\\n    "trudeau_tweet should be a Python string"\n    assert isinstance(trump_tweet_vectorized, scipy.sparse.csr.csr_matrix), \\\n    \'Make sure to transform the Trump tweet using the TF-IDF vectorizer.\'\n    assert isinstance(trudeau_tweet_vectorized, scipy.sparse.csr.csr_matrix), \\\n    \'Make sure to transform the Trudeau tweet using hte TF-IDF vectorizer.\'\n    assert trump_tweet_pred == [\'Donald J. Trump\'], \\\n    \'Your tweet was not classified as a Trump tweet, try again!\'\n    assert trudeau_tweet_pred == [\'Justin Trudeau\'], \\\n    \'Your tweet was not classified as a Trudeau tweet, try again!\'    ')

