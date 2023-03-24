
# Import statements
import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('file.csv')
docs=df.review_body
#st.dataframe(df.review_body)


model=joblib.load('review.joblib')
wo=pd.read_csv('poswords.csv')
ww=pd.read_csv('negwords.csv')
pw=wo.positive_word.to_list()
nw=ww.negative_word.to_list()
ids=set(df.product_id)
p=[]
k=[]
n = st.selectbox("Enter product id : ",ids)
def pred(n):
    if n in ids:
        
        reviews_by_product = {}
        for pid, reviews in df.groupby('product_id')['review_body']:
            if pid==n:
               reviews_by_product[pid] = list(reviews)

        # print the reviews for each product ID
        for pid, reviews in reviews_by_product.items():
            if pid==n:
                pt=df.loc[df['product_id'] == n]['product_title'].iloc[0]
                st.write('Product title : ',pt)
                st.write(f'Product ID: {pid}')
                st.write('Reviews:', reviews)
        sentiment_by_product = {}
        for pid, senti in df.groupby('product_id')['sentiment_score']:
            if pid==n:
               sentiment_by_product[pid] = list(senti)

        # print the reviews for each product ID
        for pid, senti in sentiment_by_product.items():
            if pid==n:
                st.write(f'Product ID: {pid}')
                st.write('sentiment:', senti)
                sentiment=Counter(senti).most_common(1)[0][0]
                st.write('original sentiment : ', sentiment)
        for i in range(0,len(reviews)):
            for token in str(reviews[i]).split():
                if TextBlob(token).sentiment.polarity>0:
                    p.append(token)
            for token in str(reviews[i]).split():
                if TextBlob(token).sentiment.polarity<=0:
                    k.append(token)
        
      #  st.write('pw',p)
       
       # st.write('nw',k)
        if sentiment == 1:
            st.success("We recommend this product to buy")
            st.info("Because the review of this product has positive words like:")
            st.info(p[:3])
        if sentiment == 0:
            st.error("We do not recommend this product to buy")
            st.info("Because the review of this product has negative words like:")
            st.info(k[:3])

    else:
        st.info("The product details are not found")

st.button('product id predict',on_click=pred(n))

vectorizer=TfidfVectorizer(max_features=1000)
docs = [doc for doc in docs if isinstance(doc, str)]
Xt=vectorizer.fit_transform(docs).toarray()

text=[st.text_input('write a review : ')]
pp=[]
kk=[]
def pr(text):
    st.write("orignial text : ",text)
    aa=vectorizer.transform(text)

    final_sent=model.predict(aa)
    for token in str(text[0]).split():
        if TextBlob(token).sentiment.polarity>0:
            pp.append(token)
    for token in str(text[0]).split():
        if TextBlob(token).sentiment.polarity<=0:
            kk.append(token)
    
    if TextBlob(text[0]).sentiment.polarity>0:
        st.write("predicted sentiment is : ",final_sent)
        st.success("The given text review is positive")
        st.info("Based on the below words the review is positive")
        st.info(pp[:3])
    else:
        st.error("The given text review is negative")
        st.info("Based on the below words the review is negative")
        st.info(kk[:3])

st.button('review predict',on_click=pr(text))
