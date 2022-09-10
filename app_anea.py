from dis import show_code
from http import server
from operator import le
from textwrap import shorten
import streamlit as st
import geopandas as gpd

from IPython.core.display import display, HTML

import folium
from streamlit_folium import folium_static

from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

import numpy as np
import os
import pandas as pd

st.markdown("""# Next Restaurant in Geneva
## Click on the Markers to get more information""")



# importing data to notebook

# main dataframe with decreased columns
data = pd.read_csv('data/data_combined_v1.04.csv')\
    [['place_id',
    'name',
    'price_level_combined',
    'user_ratings_total',
    'combined_rating',
    'geometry.location.lat', 'geometry.location.lng',
    'combined_main_category',
    'sub_category',
    'district',
    'district_cluster',
    'combined_main_category_2']]

# dataframe contains coordinates for district clusters
df_cluster_centers = pd.read_csv('data/data_cluster_centers_v1.02.csv').rename(columns={'district_area':'district_cluster'}) #fix later


# this is just a test on anea's idea, not the final categorization
dict_test = {
    'All':['All'],
    'European': ['All', 'European', 'French', 'Italian', 'Swiss', 'Portuguese', 'Spanish'],
    'Asian':['All', 'Japanese', 'Chinese', 'Thai', 'Indian', 'Other Asian'],
    'Middle Eastern & African': ['All', 'Lebanese', 'Turkish', 'Other Middle Eastern', 'African'],
    'American': ['All', 'American', 'South American', 'Mexican', 'Hawaiian'],
    'General': ['All', 'Restaurant', 'Bar / Pub / Bistro', 'Café'],
    'Fast Food':['All', 'Pizza', 'Hamburger', 'Chicken', 'Snacks / Take Away'],
    'Steakhouse / Barbecue / Grill': ['Steakhouse / Barbecue / Grill'],
    'Seafood': ['Seafood'],
    'Vegan / Vegetarian / Salad': ['Vegan / Vegetarian / Salad'],
    'All Other': ['All Other']}

list_district = [
    'All',
    'Saint-Jean Charmilles',
    'Bâtie - Acacias',
    'Servette Petit-Saconnex',
    'Jonction - Plainpalais',
    'Eaux-Vives - Lac',
    'Grottes Saint-Gervais',
    'Pâquis Sécheron',
    'La Cluse - Philosophes',
    'Cité-Centre',
    'Champel']


# choose the categories
st.sidebar.write('Select the type of cuisine 🍽')
rest_category_main = st.sidebar.selectbox("Select Main Restaurant Category", dict_test.keys())
rest_category = st.sidebar.selectbox("Select Sub Restaurant Category", dict_test[rest_category_main])
rest_district = st.sidebar.selectbox("Select District", list_district)

#create basic maps to be filled
geneva_1 = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4)
geneva_2 = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4, tiles='cartodbpositron')
geneva_3 = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4, tiles='cartodbpositron')
geneva_4 = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4, tiles='cartodbpositron')

def filter_data(data, rest_district, rest_category_main, rest_category):
    """
    Filters main dataframe based on district or restaurant selection
    Returns a filtered dataframe
    """
    if rest_district != 'All':
        data = data[data['district']==rest_district]

    if rest_category_main != 'All':
        data = data[data['combined_main_category_2']==rest_category_main]

    if rest_category != 'All':
        data = data[data['combined_main_category'].str.contains(rest_category)]

    return data.reset_index(drop=True)


#filter and search the restaurants
def search(df, category):
  search = lambda x:True if category.lower() in x.lower() else False
  venues = df[df['combined_main_category'].apply(search)].reset_index(drop='index')
  venues_lat_long = list(zip(venues['geometry.location.lat'], venues['geometry.location.lng']))
  return venues

data['combined_main_category'].fillna(value='not defined', inplace=True)

df = filter_data(data, rest_district, rest_category_main, rest_category)


# fill the maps


marker_cluster = folium.plugins.MarkerCluster().add_to(geneva_1)

if len(df)!=0:
    for i,row in df.iterrows():
        folium.Marker(
            location=[row['geometry.location.lat'], row['geometry.location.lng']],
            popup=[row]
            ).add_to(marker_cluster)

folium.LayerControl().add_to(geneva_1)

group0 = folium.FeatureGroup(name='<span style=\\"color: red;\\">rating below 3.0</span>')
group1 = folium.FeatureGroup(name='<span style=\\"color: orange;\\">rating between 3.0 and 4.0</span>')
group2 = folium.FeatureGroup(name='<span style=\\"color: lightgreen;\\">rating between 4 and 4.5</span>')
group3 = folium.FeatureGroup(name='<span style=\\"color: green;\\">rating above 4.5</span>')
for i,row in df.iterrows():
    if row['combined_rating']<3.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']],radius=4, color='red',fillColor='red', fill=True, popup=[row['name'], row['combined_rating']]).add_to(group0)
    elif row['combined_rating']>=3.0 and row['combined_rating']<4.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='orange',fillColor='orange', fill=True, popup=[row['name'], row['combined_rating']] ).add_to(group1)
    elif row['combined_rating']>=4.0 and row['combined_rating']<4.5:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='lightgreen',fillColor='lightgreen', fill=True, popup=[row['name'], row['combined_rating']] ).add_to(group2)
    elif row['combined_rating']>=4.5:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='green',fillColor='green', fill=True, popup=[row['name'], row['combined_rating']] ).add_to(group3)

group0.add_to(geneva_3)
group1.add_to(geneva_3)
group2.add_to(geneva_3)
group3.add_to(geneva_3)

folium.map.LayerControl('topright', collapsed=False).add_to(geneva_3)

group0 = folium.FeatureGroup(name='<span style=\\"color: red;\\">$$$$</span>')
group1 = folium.FeatureGroup(name='<span style=\\"color: orange;\\">$$$-$$$$</span>')
group2 = folium.FeatureGroup(name='<span style=\\"color: green;\\">$-$$</span>')
for i,row in df.iterrows():
    if row['price_level_combined']<3:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']],radius=4, color='green',fillColor='green', fill=True, popup=[row['name'], row['price_level_combined']]).add_to(group2)
    elif row['price_level_combined']<4 and row['price_level_combined']>=3:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='orange',fillColor='orange', fill=True, popup=[row['name'], row['price_level_combined']] ).add_to(group1)
    elif row['price_level_combined']>=4:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='red',fillColor='red', fill=True, popup=[row['name'], row['price_level_combined']] ).add_to(group0)

group0.add_to(geneva_2)
group1.add_to(geneva_2)
group2.add_to(geneva_2)
folium.map.LayerControl('topright', collapsed=False).add_to(geneva_2)


group0 = folium.FeatureGroup(name='<span style=\\"color: red;\\">less then 50 ratings</span>')
group1 = folium.FeatureGroup(name='<span style=\\"color: orange;\\">between 50 and 149 ratings</span>')
group2 = folium.FeatureGroup(name='<span style=\\"color: lightgreen;\\">between 150 and 199 ratings</span>')
group3 = folium.FeatureGroup(name='<span style=\\"color: green;\\">more then 200 rantings ratings</span>')
for i,row in df.iterrows():
    if row['user_ratings_total']<50.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']],radius=4, color='red',fillColor='red', fill=True, popup=[row['name'], row['user_ratings_total']]).add_to(group0)
    elif row['user_ratings_total']>=50.0 and row['user_ratings_total']<150.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='orange',fillColor='orange', fill=True, popup=[row['name'], row['user_ratings_total']] ).add_to(group1)
    elif row['user_ratings_total']>=150.0 and row['user_ratings_total']<200.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='lightgreen',fillColor='lightgreen', fill=True, popup=[row['name'], row['user_ratings_total']] ).add_to(group2)
    elif row['user_ratings_total']>=200.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']], radius=4, color='green',fillColor='green', fill=True, popup=[row['name'], row['user_ratings_total']] ).add_to(group3)
group0.add_to(geneva_4)
group1.add_to(geneva_4)
group2.add_to(geneva_4)
group3.add_to(geneva_4)

folium.map.LayerControl('topright', collapsed=False).add_to(geneva_4)

# display the maps
tab1, tab2, tab3, tab4 = st.tabs(["General", "Based on Price Level", "Based on Review Score", "Based on Number of Reviews"])

with tab1:
    st.header("## Check the best and worst restaurants based on general info")
    folium_static(geneva_1)

with tab2:
    st.markdown("## Check the best and worst restaurants based on Price Level")
    folium_static(geneva_2)

with tab3:
    st.header("## Check the best and worst restaurants based on Review Score")
    folium_static(geneva_3)

with tab4:
    st.header("## Check the best and worst restaurants based on Number of Reviews")
    folium_static(geneva_4)


with st.expander("See more information"):
    st.write(f'In Geneva there are **{len(df)}** restaurants belonging to the selected category')
    st.write(f'Their general review score is: {df.combined_rating.median()}📈')
    st.write(f'Their general price level is: {round(df.price_level_combined.mean())}💲')

# all these functions can be a one class

districts = data['district'].unique()


#geneva_zip_codes.fit_bounds([venues[['geometry.location.lat'][1]], venues['geometry.location.lng'][1]])

st.markdown(""" ## Some interactive graphs""")

code_df = pd.DataFrame(index=range(len(districts)),columns=['district', 'avarage price level', 'avarage review', 'number of restaurants'])

n=0
while n<len(districts):
    for i in districts:
        code_df['district'][n]=i
        code_df['avarage price level'][n]=round(data[data['district']==i]['price_level_combined'].mean(),2)
        code_df['avarage review'][n]=data[data['district']==i]['combined_rating'].median()
        code_df['number of restaurants'][n] = len(data[data['district']==i])
        n+=1

import plotly.graph_objects as go

codes_string = ([str(x) for x in districts])

fig = go.Figure(data=[
    go.Bar(name='avarage price level', x=codes_string, y=code_df['avarage price level']),
    go.Bar(name='avarage review', x=codes_string, y=code_df['avarage review']),
])

fig1 = go.Figure(data=[go.Bar(name='number of restaurant', x=codes_string, y=code_df['number of restaurants'])])

# Change the bar mode

fig.update_layout(barmode='group', title='Check the avarage price level and review score for the selected category')
st.plotly_chart(fig)


fig1.update_layout(barmode='group', title='Check number of restaurants for the selected category')
st.plotly_chart(fig1)
