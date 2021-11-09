# CVMATE: Resume Screening Application
# PROJECT DEVELOPED BY: 19DCS103 Smit Patel && 19DCS106 Yash Patel

# ----------------------------------------------------------------------------------
# Import Packages
from PIL import Image
import streamlit as st
import PyPDF2
import chardet

import os
from os import listdir
from os.path import isfile, join
from io import StringIO

import pandas as pd
from collections import Counter

import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------

image = Image.open('logo.jpeg')
st.image(image)

# ----------------------------------------------------------------------------------
# Function to read resumes from the folder one by one
mypath='C:/Users/SMIT/Desktop/CVMATE/CV' #enter your path here where you saved the resumes
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

def pdfextract(file):
    fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
    countpage = fileReader.getNumPages()
    count = 0
    text = []
    while count < countpage:    
        pageObj = fileReader.getPage(count)
        count +=1
        t = pageObj.extractText()
        print (t)
        text.append(t)
    return text
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
#function that does phrase matching and builds a candidate profile
def create_profile(file):
    text = pdfextract(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #below is the csv where we have all the keywords, you can customize your own
    with open('C:/Users/SMIT/Desktop/CVMATE/input.csv', 'rb') as f:
        result = chardet.detect(f.read())
    keyword_dict = pd.read_csv('C:/Users/SMIT/Desktop/CVMATE/input.csv', encoding=result['encoding'])
    ml_words = [nlp(text) for text in keyword_dict['Machine Learning Engineer'].dropna(axis = 0)]
    ce_words = [nlp(text) for text in keyword_dict['Cloud Engineer'].dropna(axis = 0)]
    fd_words = [nlp(text) for text in keyword_dict['Frontend Developer'].dropna(axis = 0)]
    bd_words = [nlp(text) for text in keyword_dict['Backend Developer'].dropna(axis = 0)]
    ds_words = [nlp(text) for text in keyword_dict['Data Scientist'].dropna(axis = 0)]
    android_words = [nlp(text) for text in keyword_dict['Android Developer'].dropna(axis = 0)]
    ios_words = [nlp(text) for text in keyword_dict['iOS Developer'].dropna(axis = 0)]
    network_words = [nlp(text) for text in keyword_dict['Network Engineer'].dropna(axis = 0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('ML Engineer', None, *ml_words)
    matcher.add('Cloud Engineer', None, *ce_words)
    matcher.add('Frontend Developer', None, *fd_words)
    matcher.add('Backend Developer', None, *bd_words)
    matcher.add('Data Scientist', None, *ds_words)
    matcher.add('Android Developer', None, *android_words)
    matcher.add('iOS Developer', None, *ios_words)
    matcher.add('Network Engineer', None, *network_words)
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
       
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

    return(dataf)
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------    
#code to execute/call the above functions
final_database=pd.DataFrame()
i = 0 
while i < len(onlyfiles):
    file = onlyfiles[i]
    dat = create_profile(file)
    final_database = final_database.append(dat)
    i +=1
    print(final_database)

st.write(final_database)
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
#visualize through Matplotlib
final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace = True)
final_database2.fillna(0,inplace=True)
new_data = final_database2.iloc[:,1:]
new_data.index = final_database2['Candidate Name']
output=new_data.to_csv('output.csv')
plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        label = str(j)+": " + str(new_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')
plt.show()
st.pyplot(plt)
# ----------------------------------------------------------------------------------