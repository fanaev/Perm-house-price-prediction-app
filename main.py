import streamlit as st
import pandas as pd
import catboost
import numpy as np

st.write("""
# Perm House Price Prediction App

This app redicts the ** Perm House Price **
""")
st.write("---")

st.sidebar.header('Specify input parametres')

def user_input_features():

    housetype_options = 'PG', 'HR','BR', 'IP', 'SP', 'UP', 'LP', 'MG', 'MS', 'ST'

    totalsq = st.sidebar.slider(label = 'Total square, m^2', min_value = 12.0, max_value = 107.5)
    livesq = st.sidebar.slider(label = 'Live square, m^2', min_value = 6, max_value = 81)
    kitsq = st.sidebar.slider(label = 'Kitchen square, m^2', min_value = 1, max_value = 45)
    rooms = st.sidebar.slider(label = 'Number of rooms', min_value = 1, max_value = 9)
    distance_from_center = st.sidebar.slider(label = 'Distance from central point, km', min_value = 0.086957, max_value = 24.172379)
    totalfloor = st.sidebar.slider(label = 'Total floors', min_value = 2, max_value = 26)
    housefloor = st.sidebar.slider(label = 'House floor', min_value = 1, max_value = 25)
    housetype = st.sidebar.selectbox(label = 'House type', options=housetype_options)
    house_material = st.sidebar.selectbox(label = 'House material', options=['brick', 'panel'])

    user_features = {
        'rooms': rooms, 'housefloor': housefloor, 'firstfloor': 1 if housefloor == 1 else 0, 
        'totalfloor': totalfloor, 'livesq': livesq, 'kitsq': kitsq,
        'totalsq': totalsq, 'PG': 0, 'HR': 0, 'BR': 0, 'IP': 0, 'SP': 0, 'UP': 0, 'LP': 0, 'MG': 0, 'MS': 0, 'ST': 0,
       'brick': 0, 'panel': 0, 'realtor': np.nan, 'dist_super': np.nan, 'dist_hyper': np.nan,
       'distance from center': distance_from_center}
    user_features[housetype] = 1
    user_features[house_material] = 1
    return user_features

user_data = user_input_features()
st.header('Specified Input parametres')
st.write(user_data)
st.write('---')

st.header('Prediction, rub')
model = catboost.CatBoostRegressor().load_model('models/catboost5.cbm')
st.write(model.predict(pd.DataFrame(user_data, index = [1])))