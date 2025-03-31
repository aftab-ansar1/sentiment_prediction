import streamlit as st
import pickle
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dropout, Dense
# from tensorflow.keras.layers import Flatten, Embedding, LSTM
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#loading the model 

with open ('my_model1.pkl', 'rb') as model_file:
    my_model1= pickle.load(model_file)


#Preprocessing Text

stop_words = set(stopwords.words('english'))
stop_words.discard('not')

#Preprocessing Text to pass into the tokanizer
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Removing punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words] #stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

#app page components
#header
st.header('Product Review Emmotion Predictor')
#input box
input_text = st.text_input('Enter Text', 'Enter your review and press predict button')

#preprocessing input text
processed_text = preprocess_text(input_text)

#converting input text into list to pass to the tokanizer. naming it input tokens
input_tokens = []
input_tokens.append(processed_text)


#app page output components
st.divider()
st.write('Your Review:')
st.caption(input_text)
st.divider()

#loading wordtokanizer pickle file
#word tokanizer will convert processed text into vectors
with open ('my_keras_tokens1.pkl','rb') as handle:
    word_tokanizer = pickle.load(handle)

#tokanizing the input_tokens    
input_tokens = word_tokanizer.texts_to_sequences(input_tokens)

#limiting the tokens to 100 using padding.
# padding will use the 0 to pad the tokens having less than 100 sequences after the last sequence
input_tokens = pad_sequences(input_tokens, padding='post', maxlen=100)

#app element
#predict button to process the input
predict_button = st.button('Predict')

#Predicting the emotion on scale of 0 to 1
# 0: negative, 1: Positive
emotion_dict = {0: 'Negative',
                1: 'Neutral',
                2: 'Positive'}
if predict_button:
    prediction = my_model1.predict(input_tokens)

    st.subheader(emotion_dict[prediction.argmax()])
    
    
    # import pandas as pd
    
    # history_df = pd.read_csv('history_df1.csv')
    # history_dict = {'text': input_text,
    #                     # 'Model0 Pred': pred1,
    #                     # 'Model1 Pred': pred2,
    #                     # 'Model2 Pred': pred3,
    #                     'Model3 Pred': pred4,
                        
    #                     }
    # history_dict_df = pd.DataFrame([history_dict])
    # # history_dict_df
    
    # run_history = pd.concat([history_df, history_dict_df], axis = 0)
    # run_history.to_csv('history_df1.csv', index = False)
    
    # st.dataframe(run_history)




