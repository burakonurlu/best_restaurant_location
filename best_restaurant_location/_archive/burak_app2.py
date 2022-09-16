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
from sklearn.preprocessing import MinMaxScaler


# main dataframe with decreased columns
data = pd.read_csv('../data/data_combined_v1.04.csv')\
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
df_cluster_centers = pd.read_csv('../data/data_cluster_centers_v1.02.csv').rename(columns={'district_area':'district_cluster'}) #fix later


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

rest_category_main = st.sidebar.selectbox("Select Main Restaurant Category", dict_test.keys())
rest_category = st.sidebar.selectbox("Select Sub Restaurant Category", dict_test[rest_category_main])
rest_district = st.sidebar.selectbox("Select District", list_district)

# all these functions can be a one class
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

    data = data.groupby(['district','district_cluster'])\
        [['place_id', 'user_ratings_total','combined_rating']]\
        .agg({'place_id':'count',
        'user_ratings_total':'median',
        'combined_rating':'mean'})\
        .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})

    return data.reset_index()


def merge_data(data, rest_district, rest_category_main, rest_category):
    """
    Creates a merged data set based on filtering selections and
    returns the final data set before normalization and scoring
    """
    if rest_category == 'All':
        data = filter_data(data, rest_district, rest_category_main, rest_category)
    else:
        data = filter_data(data, rest_district, 'All', 'All')\
            .merge(
                filter_data(data, rest_district, rest_category_main, rest_category)\
                    .drop(columns=['district','user_ratings_total','combined_rating']),
                how='left',
                on='district_cluster')\
            .fillna(0)
    return data

def score_data(data, rest_district, rest_category_main, rest_category):
    """
    Normalizes merged data set and create a custom scoring
    """
    # create merged data set
    df_merged = merge_data(data, rest_district, rest_category_main, rest_category)

    # normalization
    scaler = MinMaxScaler()
    cols = df_merged.drop(columns=['district','district_cluster'])
    scaler.fit(cols)
    df_score = pd.DataFrame(scaler.transform(cols), columns=cols.columns+'_norm')

    # scoring
    df_score['score'] = (1-df_score['all_restaurants_norm'])\
                        + df_score['user_ratings_total_norm']\
                        + (1-df_score['combined_rating_norm'])

    if rest_category != 'All':
        df_score['score'] += (1-df_score[f'{rest_category.lower()}_restaurants_norm'])

    # create output data_set
    df_output = pd.concat([df_merged, df_score], axis=1)\
        .merge(df_cluster_centers, how='left', on='district_cluster')
    return df_output

def pick_location(data, rest_district, rest_category_main, rest_category):
    """
    Select best / worst location based on custom scoring
    """
    if rest_district == 'All':
        n = 10
    else:
        n = 1

    best_location = score_data(data, rest_district, rest_category_main, rest_category).nlargest(n, 'score').reset_index(drop=True)
    worst_location = score_data(data, rest_district, rest_category_main, rest_category).nsmallest(n, 'score').reset_index(drop=True)

    return best_location, worst_location


# starting locating with geneva coordinates, coordinates should change when user selects a district
geneva = folium.Map(location=[46.20494053262858, 6.142254182958967], zoom_start=13.4)

# anea's code, did not change it much
best_locations = pick_location(data, rest_district, rest_category_main, rest_category)[0]
worst_locations = pick_location(data, rest_district, rest_category_main, rest_category)[1]

for i in range(len(best_locations)):
    folium.CircleMarker(
        location=[best_locations['cluster_center_lat'][i], best_locations['cluster_center_lng'][i]],
        popup=best_locations['district_cluster'][i],
        radius=10,
        color='green',
        fillColor='green',
        fill=True).add_to(geneva)

for i in range(len(worst_locations)):
    folium.CircleMarker(
        location=[worst_locations['cluster_center_lat'][i], worst_locations['cluster_center_lng'][i]],
        popup=worst_locations['district_cluster'][i],
        radius=10,
        color='red',
        fillColor='red',
        fill=True).add_to(geneva)

folium_static(geneva)
