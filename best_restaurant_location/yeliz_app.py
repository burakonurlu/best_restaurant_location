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


# main dataframe with decreased columns
df = pd.read_csv('../data/data_combined_v1.04.csv')\
    [['place_id',
    'name',
    'price_level_combined',
    'combined_rating',
    'geometry.location.lat', 'geometry.location.lng',
    'combined_main_category',
    'sub_category',
    'district',
    'district_cluster']]