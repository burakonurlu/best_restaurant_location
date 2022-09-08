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
df = pd.read_csv('../data/data_combined_v1.03.csv')\
    [['place_id',
    'name',
    'price_level_combined',
    'combined_rating',
    'geometry.location.lat', 'geometry.location.lng',
    'combined_main_category',
    'sub_category',
    'district',
    'district_cluster']]

# fix this later, cluster 8 is an orphan cluster in the middle of nowhere and contains only 1 restaurant
df = df[df['district_cluster'] != 8]

# dataframe contains total number of restaurants per district and district cluster
df_total_restaurants = df.groupby(['district','district_cluster'])[['place_id']]\
    .count()\
    .rename(columns={'place_id':'total_restaurants'})\
    .sort_values(by='district_cluster')\
    .reset_index()

# dataframe contains coordinates for district clusters
df_cluster_centers = pd.read_csv('../data/data_cluster_centers.csv').rename(columns={'district_area':'district_cluster'})


# this is just a test on anea's idea, not the final categorization
dict_test = {'General': ['Restaurant', 'Fast food / Snacks / Take Away', 'Bar / Pub / Bistro', 'Café'],
 'European': ['European', 'French', 'Italian', 'Swiss', 'Portuguese', 'Spanish'],
 'Pizza': ['Pizza'],
 'Asian':['Japanese', 'Chinese', 'Thai', 'Other Asian'],
 'American': ['American'],
 'Hamburger': ['Hamburger'],
 'Middle Eastern': ['Lebanese', 'Turkish', 'Other Middle Eastern'],
 'South American': ['South American'],
 'Indian': ['Indian'],
 'Steakhouse / Barbecue / Grill': ['Steakhouse / Barbecue / Grill'],
 'Mexican': ['Mexican'],
 'African': ['African'],
 'Seafood': ['Seafood'],
 'Vegan / Vegetarian / Salad': ['Vegan / Vegetarian / Salad'],
 'Chicken': ['Chicken'],
 'Hawaiian': ['Hawaiian'],
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

first_choice = st.sidebar.selectbox("First level options", dict_test.keys())
second_choice = st.sidebar.selectbox("Second level options", dict_test[first_choice])
select_district = st.sidebar.selectbox("Second level options", list_district)

# this can be a seperate .py file and can be reworked, it does the job for the moment
def find_locations(df, rest_district="All", rest_category="All"):
    """
    Filters main dataframe based on district or restaurant selection
    Returns best and worst locations
    """
    if rest_district == 'All' and rest_category == 'All':

        df_filtered = df

        best_locations = df_filtered.groupby(['district','district_cluster'])[['place_id']].count()\
            .rename(columns={'place_id':'total_restaurants'})\
            .nsmallest(10, 'total_restaurants', 'all')\
            .reset_index()\
            .merge(df_cluster_centers, how='left', on='district_cluster')

        worst_locations = df_filtered.groupby(['district','district_cluster'])[['place_id']].count()\
            .rename(columns={'place_id':'total_restaurants'})\
            .nlargest(10, 'total_restaurants', 'all')\
            .reset_index()\
            .merge(df_cluster_centers, how='left', on='district_cluster')

    elif rest_district == 'All':
        df_filtered = df[df['combined_main_category'].str.contains(rest_category)]

        best_locations = df_filtered.groupby('district_cluster')[['place_id']].count()\
            .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})\
            .reset_index()\
            .merge(df_total_restaurants, how='right', on='district_cluster').fillna(0)\
            .nsmallest(10, [f'{rest_category.lower()}_restaurants','total_restaurants'], 'all')\
            .merge(df_cluster_centers, how='left', on='district_cluster')

        worst_locations = df_filtered.groupby('district_cluster')[['place_id']].count()\
            .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})\
            .reset_index()\
            .merge(df_total_restaurants, how='right', on='district_cluster').fillna(0)\
            .nlargest(10, [f'{rest_category.lower()}_restaurants','total_restaurants'], 'all')\
            .merge(df_cluster_centers, how='left', on='district_cluster')

    elif rest_category == 'All':
        df_filtered = df[df['district'] == rest_district]

        best_locations = df_filtered.groupby(['district','district_cluster'])[['place_id']].count()\
            .rename(columns={'place_id':'total_restaurants'})\
            .nsmallest(1, 'total_restaurants', 'all')\
            .reset_index()\
            .merge(df_cluster_centers, how='left', on='district_cluster')

        worst_locations = df_filtered.groupby(['district','district_cluster'])[['place_id']].count()\
            .rename(columns={'place_id':'total_restaurants'})\
            .nlargest(1, 'total_restaurants', 'all')\
            .reset_index()\
            .merge(df_cluster_centers, how='left', on='district_cluster')

    else:
        df_filtered =  df[(df['district'] == rest_district) & (df['combined_main_category'].str.contains(rest_category))]

        best_locations = df_filtered.groupby('district_cluster')[['place_id']].count()\
            .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})\
            .reset_index()\
            .merge(df_total_restaurants[df_total_restaurants['district']==rest_district], how='right', on='district_cluster').fillna(0)\
            .nsmallest(1, [f'{rest_category.lower()}_restaurants','total_restaurants'], 'all')\
            .merge(df_cluster_centers, how='left', on='district_cluster')

        worst_locations = df_filtered.groupby('district_cluster')[['place_id']].count()\
            .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})\
            .reset_index()\
            .merge(df_total_restaurants[df_total_restaurants['district']==rest_district], how='right', on='district_cluster').fillna(0)\
            .nlargest(1, [f'{rest_category.lower()}_restaurants','total_restaurants'], 'all')\
            .merge(df_cluster_centers, how='left', on='district_cluster')

    return best_locations, worst_locations


# starting locating with geneva coordinates, coordinates should change when user selects a district
geneva = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4)

# default starting categories
rest_category = second_choice
rest_district = select_district

# dictionaries for dropdown menus
## rest category should be decreased later, the main purpose is to have a working version for the moment

## will be added ##

# anea's code, did not change it much
best_locations = find_locations(df, rest_district, rest_category)[0]
worst_locations = find_locations(df, rest_district, rest_category)[1]


for i in range(len(best_locations)):
    folium.CircleMarker(
        location=[best_locations['cluster_center_lat'][i], best_locations['cluster_center_lng'][i]],
        popup=best_locations['district_cluster'][i],
        radius=5,
        color='green',
        fillColor='green',
        fill=True).add_to(geneva)

for i in range(len(worst_locations)):
    folium.CircleMarker(
        location=[worst_locations['cluster_center_lat'][i], worst_locations['cluster_center_lng'][i]],
        popup=worst_locations['district_cluster'][i],
        radius=5,
        color='red',
        fillColor='red',
        fill=True).add_to(geneva)


folium_static(geneva)
