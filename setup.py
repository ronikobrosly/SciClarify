
%autoindent
import re
import csv
from Bio import Entrez
from xml.dom import minidom
import xml.dom.minidom
from textstat.textstat import textstat ## https://github.com/shivam5992/textstat
from textblob import TextBlob ## http://textblob.readthedocs.org/en/dev/
import nltk ## http://www.nltk.org
from nltk.corpus import wordnet ## http://www.nltk.org/howto/wordnet.html
from nltk.corpus import stopwords # Import the stop word list
from gensim import corpora, models, similarities ## https://radimrehurek.com/gensim/tutorial.html
import pandas as pd
import numpy as np
import MySQLdb as mdb
import sys
from sklearn.feature_extraction.text import CountVectorizer
import sklearn as sk
from collections import Counter


###########################################
## PSYCHIATRY JOURNALS 
###########################################

## Number of days ago to look for good/bad abstracts (probably need more for bad journals)
good_rel_date_num = 365
bad_rel_date_num = 3000


Entrez.email = "roni.kobrosly@gmail.com"
search_results = Entrez.read(Entrez.esearch(db="pubmed", term="am j psychiatry", reldate=good_rel_date_num, datetype="pdat", usehistory="y"))
count = int(search_results["Count"])
print("Found %i results" % count)

out_handle = open("/Users/kobrosly/desktop/getting_more_data/good_psychiatry_citations.xml", "w")
fetch_handle = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml", webenv=search_results["WebEnv"], query_key=search_results["QueryKey"])
data = fetch_handle.read()
fetch_handle.close()
out_handle.write(data)
out_handle.close()


dom = minidom.parse("/Users/kobrosly/desktop/getting_more_data/good_psychiatry_citations.xml")
foo = dom.getElementsByTagName("Abstract")
number = len(foo)

good_abstract_list = []
for study in range(number):
	if len(dom.getElementsByTagName("Abstract")[study].childNodes) == 9:
		temp = ''
		for section in range(4):
			temp = temp + dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[section].childNodes[0].data
	#else: temp = dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	else: continue
	good_abstract_list.append(temp)


GOOD_ABS = pd.Series(good_abstract_list)

df_a = pd.DataFrame(GOOD_ABS, columns=['abstract'])
df_b = pd.DataFrame(pd.Series([1]*len(GOOD_ABS)), columns=['journal'])
df_temp1 = pd.concat([df_a, df_b], join='outer', axis=1)





Entrez.email = "roni.kobrosly@gmail.com"
search_results = Entrez.read(Entrez.esearch(db="pubmed", term="Biopsychosoc Med", reldate=bad_rel_date_num, datetype="pdat", usehistory="y"))
count = int(search_results["Count"])
print("Found %i results" % count)

out_handle = open("/Users/kobrosly/desktop/getting_more_data/bad_psychiatry_citations.xml", "w")
fetch_handle = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml", webenv=search_results["WebEnv"], query_key=search_results["QueryKey"])
data = fetch_handle.read()
fetch_handle.close()
out_handle.write(data)
out_handle.close()


dom = minidom.parse("/Users/kobrosly/desktop/getting_more_data/bad_psychiatry_citations.xml")
foo = dom.getElementsByTagName("Abstract")
number = len(foo)

bad_abstract_list = []
for study in range(number):
	if len(dom.getElementsByTagName("Abstract")[study].childNodes) == 9:
		temp = ''
		for section in range(4):
			temp = temp + dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[section].childNodes[0].data
	#else: temp = dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	else: continue
	bad_abstract_list.append(temp)


BAD_ABS = pd.Series(bad_abstract_list)

df_a = pd.DataFrame(BAD_ABS, columns=['abstract'])
df_b = pd.DataFrame(pd.Series([0]*len(BAD_ABS)), columns=['journal'])
df_temp2 = pd.concat([df_a, df_b], join='outer', axis=1)


df = df_temp1.append(df_temp2)
df = df.reset_index(drop=True)
df['field'] = "Psychiatry"

del df_temp1
del df_temp2
del df_a
del df_b


df.to_csv("/Users/kobrosly/desktop/getting_more_data/psychiatry.csv", encoding='utf-8')



###########################################
## EPIDEMIOLOGY JOURNALS
###########################################


## Number of days ago to look for good/bad abstracts (probably need more for bad journals)
good_rel_date_num = 365
bad_rel_date_num = 3000


Entrez.email = "roni.kobrosly@gmail.com"
search_results = Entrez.read(Entrez.esearch(db="pubmed", term="Am J Epidemiol", reldate=good_rel_date_num, datetype="pdat", usehistory="y"))
count = int(search_results["Count"])
print("Found %i results" % count)

out_handle = open("/Users/kobrosly/desktop/getting_more_data/good_epi_citations.xml", "w")
fetch_handle = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml", webenv=search_results["WebEnv"], query_key=search_results["QueryKey"])
data = fetch_handle.read()
fetch_handle.close()
out_handle.write(data)
out_handle.close()


dom = minidom.parse("/Users/kobrosly/desktop/getting_more_data/good_epi_citations.xml")
foo = dom.getElementsByTagName("Abstract")
number = len(foo)

good_abstract_list = []
for study in range(number):
	temp = ''
	temp = temp + dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	good_abstract_list.append(temp)


GOOD_ABS = pd.Series(good_abstract_list)

df_a = pd.DataFrame(GOOD_ABS, columns=['abstract'])
df_b = pd.DataFrame(pd.Series([1]*len(GOOD_ABS)), columns=['journal'])
df_temp1 = pd.concat([df_a, df_b], join='outer', axis=1)




Entrez.email = "roni.kobrosly@gmail.com"
search_results = Entrez.read(Entrez.esearch(db="pubmed", term="J Epidemiol Glob Health", reldate=bad_rel_date_num, datetype="pdat", usehistory="y"))
count = int(search_results["Count"])
print("Found %i results" % count)

out_handle = open("/Users/kobrosly/desktop/getting_more_data/bad_epi_citations.xml", "w")
fetch_handle = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml", webenv=search_results["WebEnv"], query_key=search_results["QueryKey"])
data = fetch_handle.read()
fetch_handle.close()
out_handle.write(data)
out_handle.close()


dom = minidom.parse("/Users/kobrosly/desktop/getting_more_data/bad_epi_citations.xml")
foo = dom.getElementsByTagName("Abstract")
number = len(foo)


bad_abstract_list = []
for study in range(number):
	temp = ''
	temp = temp + dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	bad_abstract_list.append(temp)


########
bad_abstract_list = []
for study in range(number):
	if len(dom.getElementsByTagName("Abstract")[study].childNodes) == 5:
		temp = ''
		temp = temp + dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	#else: temp = dom.getElementsByTagName("Abstract")[study].getElementsByTagName("AbstractText")[0].childNodes[0].data
	else: continue
	bad_abstract_list.append(temp)
########



BAD_ABS = pd.Series(bad_abstract_list)

df_a = pd.DataFrame(BAD_ABS, columns=['abstract'])
df_b = pd.DataFrame(pd.Series([0]*len(BAD_ABS)), columns=['journal'])
df_temp2 = pd.concat([df_a, df_b], join='outer', axis=1)


df = df_temp1.append(df_temp2)
df = df.reset_index(drop=True)
df['field'] = "Epidemiology"

del df_temp1
del df_temp2
del df_a
del df_b




df.to_csv("/Users/kobrosly/desktop/getting_more_data/epidemiology.csv", encoding='utf-8')



############################################
## READ Dataset into Pandas
############################################


df = pd.read_csv("/Users/kobrosly/Desktop/datasets/all_data.csv")

############################################
## Extract all of the features
############################################



def total_length (text):
	return(len(text))

def num_sentences (text):
	sents = text.split('. ')
	#del sents[-1]
	return(len(sents))

def sentence_length (text):
	sents = text.split('. ')
	sent_lengths = [len(part) for part in sents]
	#del sent_lengths[-1]
	return(sent_lengths)

def ave_sentence_length (text):
	sents = text.split('. ')
	#del sents[-1]
	avg_len = sum(len(x.split()) for x in sents) / len(sents)
	return(avg_len)

def count_semicolon (text):
	return(text.count(';'))

def count_comma (text):
	return(text.count(', '))

def avg_word_length(text):
	words = text.split()
	return(sum(len(word) for word in words)/float(len(words)))



df['total_char_len'] = [ total_length(x) for x in df['abstract'] ]
df['num_sent'] = [ num_sentences(x) for x in df['abstract'] ]
df['avg_sent_len'] = [ ave_sentence_length(x) for x in df['abstract'] ]
df['stddev_sent_len'] = [ np.std(sentence_length(x)) for x in df['abstract'] ]
df['semicolon_count'] = [ count_semicolon(x) for x in df['abstract'] ]
df['comma_count'] = [ count_comma(x) for x in df['abstract'] ]
df['num_syllables'] = [ textstat.syllable_count(x) for x in df['abstract'] ]
df['word_count'] = [ textstat.lexicon_count(x) for x in df['abstract'] ]
df['avg_word_len'] = [ avg_word_length(x) for x in df['abstract'] ]

df['flesch_score'] = [ textstat.flesch_reading_ease(x) for x in df['abstract'] ]

# REVERSE THE ORDER OF THE FLESCH_SCORE
df['flesch_score'] = [ (82 - x) for x in df['flesch_score'] ]

df['coleman_liau_score'] = [ textstat.coleman_liau_index(x) for x in df['abstract'] ]

df['clean_abstract'] = [ re.sub("[^a-zA-Z]", " ", x) for x in df['abstract'] ]
df['clean_abstract'] = [ x.lower() for x in df['clean_abstract'] ]
df['split_words'] = [ x.split() for x in df['clean_abstract'] ]







stop_temp_list = []
for row in df['split_words']:
	temp = 0
	for word in row:
		if word in stopwords.words("english"): temp = temp + 1
	stop_temp_list.append(temp)

non_stop_temp_list = []
for row in df['split_words']:
	temp = 0
	for word in row:
		if not word in stopwords.words("english"): temp = temp + 1
	non_stop_temp_list.append(temp)

df['num_stopwords'] = stop_temp_list
#df['num_non_stopwords'] = non_stop_temp_list

#Make Remove stop words
non_stop_temp_list = []
for row in df['split_words']:
	meaningful_words = [w for w in row if not w in stopwords.words("english")] 
	non_stop_temp_list.append(meaningful_words)
df['non_stopwords_split'] = non_stop_temp_list


non_stop_temp_list = []
for row in df['non_stopwords_split']:
	count = len(set(row))
	non_stop_temp_list.append(count)
df['num_unique_nonstop_words'] = non_stop_temp_list


non_stop_temp_list = []
for row in df['non_stopwords_split']:
	count = float(len(set(row)))/ float(len(row))
	non_stop_temp_list.append(count)
df['type_token_ratio'] = non_stop_temp_list


non_stop_temp_list = []
for row in df['non_stopwords_split']:
	meaningful_words = " ".join(row)
	non_stop_temp_list.append(meaningful_words)

df['non_stopwords_together'] = non_stop_temp_list



## MEASURE SUBJECTIVITY

subjectivity_list = []
for row in df['abstract']:
	temp = row
	temp = unicode(temp, errors='replace')
	blob = TextBlob(temp)
	subjectivity_list.append(blob.sentiment.subjectivity)

df['subjectivity'] = subjectivity_list







################################
## SEND TO MYSQL
################################

clean_data = df.drop(['abstract','clean_abstract','split_words','non_stopwords_split','non_stopwords_together'], axis=1)


from pandas.io import sql
con = mdb.connect('localhost', 'Roni', 'Roni', 'SciClarify');
con.set_character_set('utf8')
#cur = con.cursor()
#cur.execute('SET NAMES utf8;') 
#cur.execute('SET CHARACTER SET utf8;')
#cur.execute('SET character_set_connection=utf8;')
#data_types = {'abstract': 'LONGBLOB', 'journal': 'INT', 'field': 'TEXT'}
clean_data.to_sql(con=con, name='final_data', if_exists='replace', flavor='mysql')
#df.to_sql(con=con, name='abstracts', if_exists='replace', flavor='mysql', dtype = data_types)
con.close()







