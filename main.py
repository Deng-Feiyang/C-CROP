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
    filename_default = ['models/model_avg_del_length.p', 'models/model_avg_ins_length.p',
                        'models/model_frac_frameshift.p', 'models/model_frac_indel_ins.p', 'models/model_indel_diversity.p']
    if not filename:
        filename = filename_default

    model_avg_del_length = pickle.load(open(filename[0], 'rb'))
    model_avg_ins_length = pickle.load(open(filename[1], 'rb'))
    model_frac_frameshift = pickle.load(open(filename[2], 'rb'))
    model_frac_indel_ins = pickle.load(open(filename[3], 'rb'))
    model_indel_diversity = pickle.load(open(filename[4], 'rb'))

    models = (model_avg_del_length, model_avg_ins_length, model_frac_frameshift, model_frac_indel_ins, model_indel_diversity)

    return models


def upload_batch_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    # Drop rows with all Null
    df = df.dropna(axis=0, how='all')
    rows = df.shape[0]

    data = data_preprocessing(df, rows)
    return df, data, rows


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


def home_page_builder(models):
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
        avg_del_length = models[0].predict(data_preprocessing(data, 1))[0]
        avg_ins_length = models[1].predict(data_preprocessing(data, 1))[0]
        frac_frameshift = models[2].predict(data_preprocessing(data, 1))[0]
        frac_indel_ins = models[3].predict(data_preprocessing(data, 1))[0]
        indel_deversity = models[4].predict(data_preprocessing(data, 1))[0]
        st.write('Fraction of indel reads with insertion: ', round(100*frac_indel_ins, 0), '%')
        st.write('Fraction of frameshift: ', round(100 * frac_frameshift, 0), '%')
        st.write('Average insertion length: ', round(avg_ins_length, 1), 'bps')
        st.write('Average deletion length: ', round(avg_del_length, 1), 'bps')
        if indel_deversity > 3.38:
            st.write('Diversity: ', round(indel_deversity, 2), ' (High)')
        else:
            st.write('Diversity: ', round(indel_deversity, 2), ' (Low)')

    st.write('---')
    st.write('Author: Deng Feiyang')
    st.write(
        '[LinkedIn](https://www.linkedin.com/in/deng-feiyang/) | [Github](https://github.com/Deng-Feiyang/C-CROP)')
    st.write('')


def batch_page_builder(models):
    st.subheader('Batch Prediction')
    st.write('')
    st.write(
        'Upload your CSV file with DNA sequences to perform batch prediction.')
    st.write('')
    # Upload CSV file for batch prediction
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.text('This will probably take a few seconds...')
    st.write(
        'Note: Every row should only contain 20-nucleotide sgRNA sequence followed by the 3-nucleotide PAM sequence in **ONE** column.')
    if uploaded_file:
        df, data, rows = upload_batch_data(
            uploaded_file)
        df.columns = ['input_sequence']
        st.write('-' * 80)
        st.write('Data preview:', df.head(10))
        st.write(
            f'Uploaded data contains **{df.shape[0]}** rows and **{df.shape[1]}** columns')
        st.write('')
        if st.button("Predict"):
            start_time = datetime.datetime.now()
            avg_del_length = models[0].predict(data)
            avg_ins_length = models[1].predict(data)
            frac_frameshift = models[2].predict(data)
            frac_indel_ins = models[3].predict(data)
            indel_deversity = models[4].predict(data)
            df['frac_indel_ins'] = frac_indel_ins
            df['frac_frameshift'] = frac_frameshift
            df['avg_ins_length'] = avg_ins_length
            df['avg_del_length'] = avg_del_length
            df['indel_deversity'] = indel_deversity
            st.write('')
            st.write('-' * 80)
            st.subheader("Output")
            prediction_time = (datetime.datetime.now() - start_time).seconds
            st.write('Prediction results preview:')
            st.write(df.head(20))
            st.text(f'Running time: {prediction_time} s')
            prediction_downloader(df)


def main():
    """C-CROP demo web app"""
    st.set_page_config(page_title='C-CROP')

    st.sidebar.title('Menu')
    choose_mode = st.sidebar.selectbox("Choose the prediction mode", [
        "Single", "Batch"])

    # Load models
    models = load_models()

    st.sidebar.header('')
    st.sidebar.header('About')
    st.sidebar.info(
        'This app is created by Deng Feiyang (deng.feiyang@outlook.com).')

    # Home page building
    if choose_mode == "Single":
        home_page_builder(models)

    # Page for batch prediction
    if choose_mode == "Batch":
        batch_page_builder(models)


if __name__ == "__main__":
    main()
