#!/usr/bin/env python
# coding: utf-8

# # TESLA 10-K and Q FILLING SENTIMENT ANALYSIS
# 
# ### - MANAVV KALRA
# ---

# In[1]:


import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup


# In[2]:


master_dict=pd.read_csv('resources/LoughranMcDonald_MasterDictionary_2018.csv')

with open('resources/Stopwords.txt' ,'r') as file:
    stopwords = file.read().lower()
stopwords_ls = stopwords.split('\n')

positive_words=list(master_dict['Word'].loc[master_dict['Positive']!=0])
positive_words=[x.lower() for x in positive_words]
negative_words=list(master_dict['Word'].loc[master_dict['Negative']!=0])
negative_words=[x.lower() for x in negative_words]
uncertainty_words=list(master_dict['Word'].loc[master_dict['Uncertainty']!=0])
uncertainty_words=[x.lower() for x in uncertainty_words]
constraining_words=list(master_dict['Word'].loc[master_dict['Constraining']!=0])
constraining_words=[x.lower() for x in constraining_words]
litigious_words=list(master_dict['Word'].loc[master_dict['Litigious']!=0])
litigious_words=[x.lower() for x in litigious_words]


# In[3]:


df=pd.read_excel('resources/tesla.xlsx')
df=df.iloc[:28,:] #LAST FIVE YEARS
df.head()


# In[4]:


def replace_url(txt):
    txt=txt.replace('-index.htm','.txt')
    return txt


# In[5]:


df['Filings URL']=df['Filings URL'].map(replace_url)


# In[6]:


#The 3 lists
mda_ls=[]
qqdmr_ls=[]
rf_ls=[]


# In[7]:


#Regular expressions

#HTML tags
html= r"<[^>]*>"

#Management's Discussion and Analysis
mda = r"item\s*\d\.\s*Management\'s Discussion and Analysis.*?item\s*\d\(?[a-z]?\)?\.\s*"

#Quantitavite and Qualitative Disclosures about Market Risk
qqdmr = r"item\s*\d\(?[a-z]?\)?\.\s*Quantitative and Qualitative Disclosures about Market Risk.*?item\s*\d\(?[a-z]?\)?\.\s*"
    
#Risk Factors
rf = r"item\s*\d\(?[a-z]?\)?\.\s*Risk Factors.*?item\s*\d\(?[a-z]?\)?\.\s*"


# In[8]:


#clean(txt) and prep(url) functions

def clean(txt):
    txt = txt.strip()
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('&nbsp;', ' ').replace('\xa0',' ')
    return txt

def prep(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    txt = str(soup)
    txt = re.sub(html,' ',txt) #remove HTML tags
    all_ls=[[mda,mda_ls],[qqdmr,qqdmr_ls],[rf,rf_ls]]
    for regex,ls in all_ls:
        check=re.findall(regex, txt, re.I | re.M | re.DOTALL)
        if check:
            ls.append(clean(max(check, key=len)))
        else:
            ls.append('')
    return None


# In[9]:


df['Filings URL'].apply(prep)


# In[10]:


#For tokenizing (Words and sentences)
from nltk.tokenize import RegexpTokenizer,sent_tokenize


# In[11]:


#REMOVE STOP WORDS
def remove_stopwords(tokens):
    return list(filter(lambda x: x not in stopwords_ls, tokens))

#CONVERT THE TEXT INTO TOKENS
def tokenize(txt):
    txt = txt.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(txt)
    return remove_stopwords(tokens)

#CALCULATE POSITIVE WORD SCORE
def calc_positive_score(txt):
    n_positive=0
    tokens=tokenize(txt)
    for token in tokens:
        if token in positive_words:
            n_positive+=1
    return n_positive

#CALCULATE NEGATIVE WORD SCORE
def calc_negative_score(txt):
    n_negative=0
    tokens=tokenize(txt)
    for token in tokens:
        if token in negative_words:
            n_negative-=1
    negative_score=-1*n_negative
    return negative_score

#CALCULATE POLARITY SCORE
def calc_polarity_score(positive_score,negative_score):
    polarity_score=(positive_score-negative_score)/((positive_score+negative_score)+0.000001)
    return polarity_score

#CALCULATE AVERAGE SENTENCE LENGTH
def calc_avg_sentence_length(txt):
    avg_sentence_length=0
    sentences = sent_tokenize(txt)
    tokens = tokenize(txt)
    if len(sentences) != 0:
        avg_sentence_length=len(tokens)/len(sentences)
    return avg_sentence_length

#CALCULATE WORD COUNT
def calc_wordcount(txt):
    n_words=len(tokenize(txt))
    return n_words

#CALCULATE COMPLEX WORD COUNT
def calc_complexword_count(txt):
    tokens = tokenize(txt)
    n_complexwords = 0
    vowels = 'aeiou'
    for token in tokens:
        n_syllables = 0
        if token.endswith(('es','ed')):
            pass
        if token[0] in vowels:
            n_syllables+=1
        for i in range(1,len(token)):
            if token[i] in vowels and token[i-1] not in vowels:
                n_syllables+=1
        if n_syllables==0:
            n_syllables+=1
        if n_syllables>2:
            n_complexwords+=1
    return n_complexwords

#CALCULATE COMPLEX WORD PERCENTAGE
def calc_percentage_complexwords(txt):
    percentage_complexwords=0
    n_complexwords=calc_complexword_count(txt)
    n_words=calc_wordcount(txt)
    if n_words!=0:
        percentage_complexwords=n_complexwords/n_words
    return percentage_complexwords
    
#CALCULATE FOG INDEX    
def calc_fog_index(avg_sentence_length,percentage_complexwords):
    fog_index=0.4 * (avg_sentence_length + percentage_complexwords)
    return fog_index

#CALCULATE UNCERTAINTY SCORE
def calc_uncertainty_score(txt):
    uncertainty_score=0
    tokens=tokenize(txt)
    for token in tokens:
        if token in uncertainty_words:
            uncertainty_score+=1
    return uncertainty_score

#CALCULATE CONSTRAINING SCORE
def calc_constraining_score(txt):
    constraining_score=0
    tokens=tokenize(txt)
    for token in tokens:
        if token in constraining_words:
            constraining_score+=1
    return constraining_score

#CALCULATE LITIGIOUS SCORE
def calc_litigious_score(txt):
    litigious_score=0
    tokens=tokenize(txt)
    for token in tokens:
        if token in litigious_words:
            litigious_score+=1
    return litigious_score

#CALCULATE POSITIVE WORD PROPORTION
def calc_positiveword_proportion(positive_score,n_words):
    positiveword_proportion=0
    if n_words!=0:
        positiveword_proportion=positive_score/n_words
    return positiveword_proportion

#CALCULATE NEGATIVE WORD PROPORTION
def calc_negativeword_proportion(negative_score,n_words):
    negativeword_proportion=0
    if n_words!=0:
        negativeword_proportion=negative_score/n_words
    return negativeword_proportion

#CALCULATE UNCERTAINTY WORD PROPORTION
def calc_uncertaintyword_proportion(uncertainty_score,n_words):
    uncertaintyword_proportion=0
    if n_words!=0:
        uncertaintyword_proportion=uncertainty_score/n_words
    return uncertaintyword_proportion

#CALCULATE CONSTRAINING WORD PROPORTION
def calc_constrainingword_proportion(constraining_score,n_words):
    constrainingword_proportion=0
    if n_words!=0:
        constrainingword_proportion=constraining_score/n_words
    return constrainingword_proportion

#CALCULATE LITIGIOUS WORD PROPORTION
def calc_litigiousword_proportion(litigious_score,n_words):
    litigiousword_proportion=0
    if n_words!=0:
        litigiousword_proportion=litigious_score/n_words
    return litigiousword_proportion


# In[12]:


#Applying the functions to all three lists and storing the results in the dataframe

apply_ls=[['mda',mda_ls],['qqdmr',qqdmr_ls],['rf',rf_ls]]

for section,ls in apply_ls:
    df[section+'_positive_score'] = list(map(calc_positive_score,ls))
    df[section+'_negative_score'] = list(map(calc_negative_score,ls))
    df[section+'_polarity_score'] = np.vectorize(calc_polarity_score)(df[section+'_positive_score'],df[section+'_negative_score'])
    df[section+'_average_sentence_length'] = list(map(calc_avg_sentence_length,ls))
    df[section+'_percentage_of_complex_words'] = list(map(calc_percentage_complexwords,ls))
    df[section+'_fog_index'] = np.vectorize(calc_fog_index)(df[section+'_average_sentence_length'],df[section+'_percentage_of_complex_words'])
    df[section+'_complex_word_count'] = list(map(calc_complexword_count,ls))    
    df[section+'_word_count'] = list(map(calc_wordcount,ls))
    df[section+'_uncertainty_score'] = list(map(calc_uncertainty_score,ls))
    df[section+'_constraining_score'] = list(map(calc_constraining_score,ls))
    df[section+'_litigious_score'] = list(map(calc_litigious_score,ls))
    df[section+'_positive_word_proportion'] = np.vectorize(calc_positiveword_proportion)(df[section+'_positive_score'],df[section+'_word_count'])
    df[section+'_negative_word_proportion'] = np.vectorize(calc_negativeword_proportion)(df[section+'_negative_score'],df[section+'_word_count'])
    df[section+'_uncertainty_word_proportion'] = np.vectorize(calc_uncertaintyword_proportion)(df[section+'_uncertainty_score'],df[section+'_word_count'])
    df[section+'_constraining_word_proportion'] = np.vectorize(calc_constrainingword_proportion)(df[section+'_constraining_score'],df[section+'_word_count'])
    df[section+'_litigious_word_proportion'] = np.vectorize(calc_litigiousword_proportion)(df[section+'_litigious_score'],df[section+'_word_count'])


# In[13]:


df.to_excel('tesla_result.xlsx')


# ## FIN
# ---
