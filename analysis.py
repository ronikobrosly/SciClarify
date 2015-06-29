
%autoindent
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sys
import patsy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from textstat.textstat import textstat ## https://github.com/shivam5992/textstat
from textblob import TextBlob ## http://textblob.readthedocs.org/en/dev/
import nltk ## http://www.nltk.org
from nltk.corpus import wordnet ## http://www.nltk.org/howto/wordnet.html
from nltk.corpus import stopwords # Import the stop word list
import re


#############################################
# FETCH DATA 
#############################################

df = pd.read_csv('/Users/kobrosly/Desktop/final_data.csv')

field_input = "Psychiatry"
specific_data = df[df.field == field_input]


#############################################
# RUN THE MODEL
#############################################

# create dataframes with an intercept column 
y, X = patsy.dmatrices('journal ~ total_char_len + num_sent + avg_sent_len + stddev_sent_len + \
                  semicolon_count + comma_count + num_syllables + word_count + avg_word_len + \
                  flesch_score + coleman_liau_score + \
                  num_stopwords + num_unique_nonstop_words + \
                  type_token_ratio + subjectivity',
                  specific_data, return_type="dataframe")

# flatten y into a 1-D array
y = np.ravel(y)

#Fit the model
model = RandomForestClassifier(random_state = 998)
model = model.fit(X, y)




#############################################
# CALCULATE THE SCORE
#############################################


## CUSTOM FUNCTIONS

def total_length (text):
  return(len(text))

def num_sentences (text):
  sents = text.split('. ')
  return(len(sents))

def sentence_length (text):
  sents = text.split('. ')
  sent_lengths = [len(part) for part in sents]
  return(sent_lengths)

def ave_sentence_length (text):
  sents = text.split('. ')
  avg_len = sum(len(x.split()) for x in sents) / len(sents)
  return(avg_len)

def count_semicolon (text):
  return(text.count(';'))

def count_comma (text):
  return(text.count(', '))

def avg_word_length(text):
  words = text.split()
  return(sum(len(word) for word in words)/float(len(words)))


## WARNING MESSAGE FLAG, IF THIS BECOMES 1, WEBSITE REPORTS ERROR
warning_message = 0


# total_char_len
try:
  total_char_len = total_length(AB) 
except:
  warning_message = 1

# num_sent
try:
  num_sent = num_sentences(AB)  
except: 
  warning_message = 1

# avg_sent_len
try:
  avg_sent_len = ave_sentence_length(AB) 
except:
  warning_message = 1

# stddev_sent_len
try:
  stddev_sent_len = np.std(sentence_length(AB)) 
except: 
  warning_message = 1

# semicolon_count
try:
  semicolon_count = count_semicolon(AB) 
except:
  warning_message = 1

# comma_count
try:
  comma_count = count_comma(AB) 
except:
  warning_message = 1

# num_syllables
try: 
  num_syllables = textstat.syllable_count(AB)
except: 
  warning_message = 1

# word_count
try:
  word_count = textstat.lexicon_count(AB) 
except: 
  warning_message = 1

# avg_word_len
try: 
  avg_word_len = avg_word_length(AB) 
except:
  warning_message = 1
  
# flesch_score
try: 
  flesch_score = textstat.flesch_reading_ease(AB) 
except:
  warning_message = 1

# coleman_liau_score
try:
  coleman_liau_score = textstat.coleman_liau_index(AB) 
except: 
  warning_message = 1

# num_stopwords
try:
  clean_abstract = re.sub("[^a-zA-Z]", " ", AB)
  clean_abstract = clean_abstract.lower() 
  split_words = clean_abstract.split()
  stops = set(stopwords.words("english"))
  num_stopwords = len([w for w in split_words if w in stops]) 
except:
  warning_message = 1

# num_unique_nonstop_words
try:
  meaningful_words = [w for w in split_words if not w in stopwords.words("english")] 
  num_unique_nonstop_words = len(meaningful_words) 
except:
  warning_message = 1

# type_token_ratio
try:
  type_token_ratio = float(len(set(meaningful_words)))/ float(len(meaningful_words)) 
except:
  warning_message = 1

# subjectivity
try:
  temp = unicode(AB, errors='replace')
  blob = TextBlob(temp)
  subjectivity = (blob.sentiment.subjectivity) 
except: 
  warning_message = 1




# FIRST VALUE IS IRRELVANT, MAKE IT ZERO
x_test = [1, total_char_len, num_sent, avg_sent_len, stddev_sent_len, semicolon_count, comma_count, num_syllables, word_count, avg_word_len, flesch_score, coleman_liau_score, num_stopwords, num_unique_nonstop_words, type_token_ratio, subjectivity]

## PROBABILITY OF CLASSIFICATION AS GOOD JOURNAL
round(model.predict_proba(x_test)[0][1], 3) * 100





########################################
## PROVIDE SPECIFIC RECOMMENDATIONS
########################################

def recommendations(abstract):
  recommend_text = ""
  if total_char_len < np.percentile((specific_data[specific_data.journal == 1])['total_char_len'], 30): recommend_text += "Your abstract has very little text!\n"
  if total_char_len > np.percentile((specific_data[specific_data.journal == 1])['total_char_len'], 70): recommend_text += "Your abstract has so much text!\n"
  #
  if num_sent < np.percentile((specific_data[specific_data.journal == 1])['num_sent'], 40): recommend_text += "Your text doesn't have many sentences. Consider adding more sentences or breaking up your long ones.\n"
  if num_sent > np.percentile((specific_data[specific_data.journal == 1])['num_sent'], 60): recommend_text += "Your text has lots of sentences. Consider combining some of your sentences.\n"
  #
  if avg_sent_len < np.percentile((specific_data[specific_data.journal == 1])['avg_sent_len'], 40): recommend_text += "Your average sentence length is a little low.\n"
  if avg_sent_len > np.percentile((specific_data[specific_data.journal == 1])['avg_sent_len'], 60): recommend_text += "Your average sentence length is high.\n"
  #
  if stddev_sent_len > np.percentile((specific_data[specific_data.journal == 1])['stddev_sent_len'], 80): recommend_text += "The length of your sentences varies wildly. Try to make them more equal in length.\n"
  #
  if semicolon_count > np.percentile((specific_data[specific_data.journal == 1])['semicolon_count'], 75): recommend_text += "You are using way too many semicolons!\n"
  #
  if comma_count > np.percentile((specific_data[specific_data.journal == 1])['comma_count'], 75): recommend_text += "You are using way too many commas! Consider breaking up some of those sentences.\n"
  #
  if num_syllables > np.percentile((specific_data[specific_data.journal == 1])['num_syllables'], 75): recommend_text += "Compared with published abstracts, you use way more syllables.\n"
  #
  if flesch_score < np.percentile((specific_data[specific_data.journal == 1])['flesch_score'], 30): recommend_text += "Your text's Flesch Reading Score indicates your abstract is difficult to read and potentially too jargony.\n"
  #
  if num_stopwords > np.percentile((specific_data[specific_data.journal == 1])['num_stopwords'], 75): recommend_text += "Your text has too many stop words (e.g. and, but, also). Consider editting some of them out.\n"
  #
  if num_unique_nonstop_words < np.percentile((specific_data[specific_data.journal == 1])['num_unique_nonstop_words'], 30): recommend_text += "The number of unique words in your text is low. Try to use a wider range of words or increasing your total number of words if your text is short.\n"
  #
  if subjectivity > np.percentile((specific_data[specific_data.journal == 1])['subjectivity'], 75): recommend_text += "Your text is reading a little too subjective. Please try to make your statements more objective.\n"
  #
  print recommend_text

recommendations(AB)
