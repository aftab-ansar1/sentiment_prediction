import streamlit as st
import pickle
import nltk
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

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


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
tab1, tab2 = st.tabs(["App", "About"])
with tab1:
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
    
    with tab2:
        st.subheader("About App")
        st.write("The app is designed to help to understand the machine of the emotion expressed in the review of a product or text input of the user")
        st.write("The text input need not be a product review but since, the model is trained on the product review data, it works more accurate with the same kind of information")
        st.write("The model interprets the emotions as Positive, Negative and Neutral.")

        st.write("The emotions interpreted by the model need not to be accurate due to the limitation of the training data and machine learning capability.")
        st.write("The interpretations may be affected by the complexity of the sentences, tone, satire, irony, context or references to some historical or classical events/enteties.")

        st.write("User descrition is expected while using this app.")
        
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




