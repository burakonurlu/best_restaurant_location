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

st.markdown("""# Next Restaurant in Geneva
## Click on the Markers to get more information""")

# importing data to notebook

df=pd.read_csv('../data/data_combined_v1.03.csv')


geneva=folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=11)



level_two_options = {'General / Restaurant': ['General'],
 'European': ['European', 'French', 'Italian', 'Swiss', 'Portuguese', 'Spanish'],
 'French': [],
 'Italian': ['Italian', 'Pizza'],
 'General / Fast food / Snacks / Take Away': ['General'],
 'Pizza': [],
 'Swiss': [],
 'General / Bar / Pub / Bistro': ['General'],
 'Japanese': [],
 'General / Café': ['General'],
 'Asian': ['Thai', 'Chinese', 'Other Asian'],
 'Thai': [],
 'Chinese': [],
 'American': ['American', 'Hamburger'],
 'Hamburger': [],
 'Lebanese': ['Middle Eastern'],
 'Other Asian': [],
 'Turkish': ['Middle Eastern'],
 'South American': ['South American'],
 'Indian': ['Indian'],
 'Steakhouse / Barbecue / Grill': ['Steakhouse / Barbecue / Grill'],
 'Mexican': ['Mexican'],
 'Portuguese': [],
 'African': ['African'],
 'Seafood': ['Seafood'],
 'Spanish': [],
 'Vegan / Vegetarian / Salad': ['Vegan / Vegetarian / Salad'],
 'Other Middle Eastern': ['Middle Eastern'],
 'Chicken': ['Chicken'],
 'Hawaiian': ['Hawaiian'],
 'All Other': ['All Other']
}

first_choice = "General"
first_choice = st.sidebar.selectbox("First level options", level_two_options.keys())
#second_choice = st.sidebar.selectbox("Second level options", level_two_options[first_choice])

def search(df, category):
  search = lambda x:True if category.lower() in x.lower() else False
  venues = df[df['combined_main_category'].apply(search)].reset_index(drop='index')
  venues_lat_long = list(zip(venues['geometry.location.lat'], venues['geometry.location.lng']))
  return venues

df['combined_main_category'].fillna(value='not defined', inplace=True)
if level_two_options[first_choice]==[]:
    df = search(df, first_choice)
else:
    df = search(df, second_choice)

for i in range(len(df)):
    if df['combined_rating'][i]<4.0:
        folium.CircleMarker(location=[df['geometry.location.lat'][i], df['geometry.location.lng'][i]],radius=5, color='red',fillColor='red', fill=True, popup=[df['name'][i]]).add_to(geneva)
    elif df['combined_rating'][i]>=4.0:
        folium.CircleMarker(location=[df['geometry.location.lat'][i], df['geometry.location.lng'][i]], radius=5, color='green',fillColor='green', fill=True, popup=[df['name'][i]] ).add_to(geneva)
    else:
        folium.CircleMarker(location=[df['geometry.location.lat'][i], df['geometry.location.lng'][i]],radius=5, color='white',fillColor='white', fill=True, popup=[df['name'][i]]).add_to(geneva)

folium_static(geneva)

st.markdown("""## General Information""")

st.write(f'In Geneva there are **{len(df)}** restaurants belonging to the selected category')
st.write(f'Their general review score is: {df.combined_rating.median()}')
st.write(f'Their general price level is: {round(df.price_level_combined.mean())}')


st.markdown(""" ## If you want to dive deeper check the specific restaurant in each district""")

geneva_zip_codes=folium.Map(location=[46.20918858309737, 6.1298195041608325], zoom_start=11)

codes = df['district'].unique()

code = st.selectbox("Select district", codes)


venues = df[df['district']==code]

for i in range(len(venues)):
    folium.CircleMarker(location=[df['geometry.location.lat'][i], df['geometry.location.lng'][i]],radius=5, color='green', fillColor='green', fill=True, popup=[df['name'][i]]).add_to(geneva_zip_codes)

folium_static(geneva_zip_codes)

st.markdown(""" ## some graphs""")

code_df = pd.DataFrame(index=range(len(codes)),columns=['district', 'avarage price level', 'avarage review', 'number of restaurants'])

n=0
while n<len(codes):
    for i in codes:
        code_df['district'][n]=i
        code_df['avarage price level'][n]=round(df[df['district']==i]['price_level_combined'].mean(),2)
        code_df['avarage review'][n]=df[df['district']==i]['combined_rating'].median()
        code_df['number of restaurants'][n] = len(df)
        n+=1


import plotly.graph_objects as go

codes_string = ([str(x) for x in codes])

fig = go.Figure(data=[
    go.Bar(name='avarage price level', x=codes_string, y=code_df['avarage price level']),
    go.Bar(name='avarage review', x=codes_string, y=code_df['avarage review']),
    go.Bar(name='number of restaurant', x=codes_string, y=code_df['avarage review'])
])
# Change the bar mode
fig.update_layout(barmode='group')
st.plotly_chart(fig)
