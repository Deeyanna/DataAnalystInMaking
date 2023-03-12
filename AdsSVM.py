import streamlit as st
import pandas as pd
import pickle

st.write("""
# Sales Prediction based on Advertisment Budget App
""")

st.sidebar.header('Please select the value for Advetising Medium's Budget')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 10.0)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 5.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 6.9, 0.6)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Selection for Each Advertising Medium's Budget')
st.write(df)

loaded_model = pickle.load(open("AdvertisingSVM.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('!!Sales Prediction!!')
st.write(prediction)
