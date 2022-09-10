from http import server
from operator import le
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import plotly.express as px
import numpy as np
import os
import pandas as pd


#streamlit trials


# streamlit preparation
st.title('Select Model Weights')

# Session state callbacks
def callback_x1():
    """The callback renormalizes the values that were not updated."""
    # Here, if we modify X1, we calculate the remaining value for normalisation of X2, X3 and X4
    remain = 10 - st.session_state.x1
    # This is the proportions of X2, X3 and X4 in remain
    sum = st.session_state.x2 + st.session_state.x3
    # This is the normalisation step
    st.session_state.x2 = round(st.session_state.x2/sum*remain)
    st.session_state.x3 = round(st.session_state.x3/sum*remain)

def callback_x2():
    remain = 10 - st.session_state.x2
    sum = st.session_state.x1 + st.session_state.x3
    st.session_state.x1 = round(st.session_state.x1/sum*remain)
    st.session_state.x3 = round(st.session_state.x3/sum*remain)
def callback_x3():
    remain = 10 - st.session_state.x3
    sum = st.session_state.x1 + st.session_state.x2
    st.session_state.x1 = round(st.session_state.x1/sum*remain)
    st.session_state.x2 = round(st.session_state.x2/sum*remain)

# Sessions tate initialise
# Check if 'key' already exists in session_state
# If not, then initialize it
if 'x1' not in st.session_state:
    st.session_state['x1'] = 4

if 'x2' not in st.session_state:
    st.session_state['x2'] = 3

if 'x3' not in st.session_state:
    st.session_state['x3'] = 3

# The four sliders
st.slider("Competition, %",
                min_value = 0,
                max_value = 10,
                step=1,
                key='x1', on_change=callback_x1)

st.slider("District Popularity, %",
                min_value = 0,
                max_value = 10,
                step=1,
                key='x2', on_change=callback_x2)

st.slider("Customer Satisfaction, %",
                min_value = 0,
                max_value = 10,
                step=1,
                key='x3', on_change=callback_x3)
