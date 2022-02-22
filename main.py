import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(content):
    content = content.lower()
    content = nltk.word_tokenize(content)

    y = []
    for i in content:
        if i.isalnum():
            y.append(i)

    content = y[:]
    y.clear()

    for i in content:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    content = y[:]
    y.clear()

    for i in content:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Abusive Email Classification')

input_text = st.text_area("Enter the text")

if st.button('Predict'):
    # 1. preprocess
    transformed_content = transform_text(input_text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_content])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("Abusive")
    else:
        st.header("Non Abusive")