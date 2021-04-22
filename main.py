import streamlit as st
import pandas as pd
import pickle
import numpy as np
import datetime
import time
import lightgbm
from lightgbm import LGBMRegressor
import xgboost
import os
import base64

from PIL import Image

@st.cache
def load_models(filename=None):
    filename_default = 'model_avg_del_length.p'
    if not filename:
        filename = filename_default

    pickle_in = open(filename_default, 'rb')
    model_avg_del_length = pickle.load(pickle_in)
    # model_avg_del_length = lightgbm.Booster(model_file=filename)

    return model_avg_del_length


def upload_batch_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    # Drop rows with all Null
    df = df.dropna(axis=0, how='all')
    rows = df.shape[0]

    data = data_preprocessing(df, rows)
    return df, data, 'Uploaded file', rows


def data_preprocessing(df, rows):
    """Return encoded DNA sequences."""
    sequence_encoded = np.zeros((rows, 23, 4), dtype=bool)
    for i, val in enumerate(df.iloc[:, 0]):
        for ind, basepair in enumerate(val):
            sequence_encoded[i, ind, one_hot_index(basepair)] = 1

    sequence_array = np.empty(shape=[rows, 92], dtype=bool)
    for i, val in enumerate(sequence_encoded):
        sequence_array[i] = np.reshape(sequence_encoded[i], (1, -1))

    return sequence_array


def one_hot_index(nucleotide):
    if nucleotide == 'g':
        nucleotide = 'G'
    elif nucleotide == 'a':
        nucleotide = 'A'
    elif nucleotide == 'c':
        nucleotide = 'C'
    elif nucleotide == 't':
        nucleotide = 'T'
    nucleotide_array = ['A', 'C', 'G', 'T']
    return nucleotide_array.index(nucleotide)


def single_predictor(data, model):
    avg_del_length = model.predict(data)
    return avg_del_length


def prediction_downloader(data):
    st.write('')
    st.subheader('Download the prediction results')
    csv = data.to_csv(index=False)
    # Some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (Right-click and save as &lt;file_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)


def home_page_builder(model):
    st.title("C-CROP")
    st.write('**C**RISPR-**C**AS9 **R**epair **O**utcome **P**rediction')
    st.write('')
    st.subheader('INTRODUCTION')
    image = Image.open('crispr_banner.png')
    st.image(image, caption='', width=600)
    st.write('')
    st.write(
        'C-CROP is a machine learning tool that predicts the DNA repair outcome in CRISPR-CAS9 experiments.')
    st.write('')
    st.write(
        'Input the 20-nucleotide sgRNA sequence followed by the 3-nucleotide PAM sequence.')
    st.write(
        '(e.g., CCACCAAAGTACGATGTGAGAGG)')
    st.write('')
    st.write('')

    # Insert field to receive user inputs
    sequence = st.text_input("", "", max_chars=23, help='Input sequence...')
    if st.button("Predict"):
        st.subheader("Output")
        data = pd.DataFrame([sequence])
        avg_del_length = model.predict(data_preprocessing(data, 1))[0]
        st.write('Average deletion length:', round(avg_del_length, 1), 'bps')

    st.write('---')
    st.write('Author: Deng Feiyang')
    st.write(
        '[LinkediIn](https://www.linkedin.com/in/deng-feiyang/) | [Github](https://github.com/Deng-Feiyang/C-CROP)')
    st.write('')


def batch_page_builder(model):
    st.subheader('Batch Prediction')
    st.write('')
    st.write(
        'Upload your CSV file with DNA sequences to perform batch prediction.')
    st.write('')
    # Upload CSV file for batch prediction
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.text('This process probably takes few seconds...')
    st.write(
        'Note: Every row should only contain 20-nucleotide sgRNA sequence followed by the 3-nucleotide PAM sequence in **ONE** column.')
    if uploaded_file:
        df, data, filename, rows = upload_batch_data(
            uploaded_file)
        st.write('-' * 80)
        st.write('Uploaded data:', df.head(10))
        st.write(
            f'Uploaded data includes **{df.shape[0]}** rows and **{df.shape[1]}** columns')
        st.write('')
        if st.button("Predict"):
            start_time = datetime.datetime.now()
            avg_del_length = model.predict(data)
            avg_del_length = pd.DataFrame(avg_del_length)
            st.write('')
            st.write('-' * 80)
            st.subheader("Output")
            prediction_time = (datetime.datetime.now() - start_time).seconds
            st.write('Average deletion length:')
            st.write(avg_del_length.head(10))
            st.text(f'Running time: {prediction_time} s')
            prediction_downloader(avg_del_length)


def main():
    """C-CROP demo web app"""
    st.set_page_config(page_title='C-CROP')

    st.sidebar.title('Menu')
    choose_mode = st.sidebar.selectbox("Choose the prediction mode", [
        "Single", "Batch"])

    # Load models
    model_avg_del_length = load_models()

    st.sidebar.header('')
    st.sidebar.header('About')
    st.sidebar.info(
        'This app is created by Deng Feiyang (deng.feiyang@outlook.com).')

    # Home page building
    if choose_mode == "Single":
        home_page_builder(model_avg_del_length)

    # Page for batch prediction
    if choose_mode == "Batch":
        batch_page_builder(model_avg_del_length)


if __name__ == "__main__":
    main()

