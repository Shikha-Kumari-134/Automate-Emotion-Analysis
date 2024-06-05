# import all the required libraries and modules
import numpy as np
import pandas as pd
import pandas as pd
import nltk
import sklearn
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')
import string
punctuations=list(string.punctuation)
stop=stop+punctuations
import re
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import seaborn as sns
from nltk.stem import PorterStemmer
import pickle

#Read the Data
df=pd.read_csv('tweet_emotions.csv')

#Drop unwanted columns
df=df.drop(columns=['tweet_id'])

#Drop Duplicates
index=df[df['content'].duplicated()==True].index
df.drop(index, axis = 0, inplace = True)
df.reset_index(inplace=True, drop = True)

# Frequency distribution of'sentiment'
frequency_counts = df['sentiment'].value_counts()

frequency_percentage = (frequency_counts / len(df['sentiment'])) * 100
frequency_df = pd.DataFrame({'Counts': frequency_counts, 'Percentage': frequency_percentage})

# tokenize the string and convert into matrix
tokenizer = Tokenizer(num_words=2000, split=" ")
tokenizer.fit_on_texts(df['content'].values)

X= tokenizer.texts_to_sequences(df['content'].values)
X = pad_sequences(X)

# one hot encoding the labels
Y = pd.get_dummies(df['sentiment']).values
#  divide into training and testing data
X_train,X_test,Y_train,Y_test = sklearn.model_selection.train_test_split(X,Y,random_state=1)

def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

df['word count'] = df['content'].apply(no_of_words)

def remove_special_characters(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove special characters using regular expressions, keeping only alphanumeric characters
    clean_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]

    # Join the words back into a sentence
    clean_text = ' '.join(clean_words)
    return clean_text

# Apply the function to the 'text' column in the DataFrame
df['content'] = df['content'].apply(remove_special_characters)

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


df.content = df['content'].apply(lambda x: stemming(x))

df['word count'] = df['content'].apply(no_of_words)

#Model Development

model = Sequential()
model.add(Embedding(2000, 256, input_length=X_train.shape[1]))
model.add(Dropout(0.2))  # Adjusted dropout rate
model.add(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))  # Reduced LSTM units
model.add(Dropout(0.2))  # Adjusted dropout rate
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))  # Reduced LSTM units
model.add(Dense(13, activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train the model on training data
batch_size = 80
epochs = 15

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)



pickle.dump(model,open("model.pkl" , "wb"))

