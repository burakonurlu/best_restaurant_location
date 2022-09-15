from dis import show_code
from http import server
from operator import le
from textwrap import shorten
import streamlit as st
st.set_page_config(layout="centered", page_title="Next Resturant in Geneva", page_icon=":cook:")
import geopandas as gpd
from IPython.core.display import display, HTML
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import numpy as np
import os
import pandas as pd
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
# let's remove unused modules later

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

# Dataframe contains coordinates for district and district clusters
df_cluster_centers = pd.read_csv('data/data_cluster_centers_v1.02.csv')
df_district = pd.read_csv('data/data_district.csv')

# Functions Start
def filter_data(data, rest_district, rest_category_main, rest_category):
    """
    Filters main dataframe based on district or restaurant selection
    FOR DROWDOWN MENUS
    Returns a filtered dataframe
    """
    if rest_district != 'All':
        data = data[data['district']==rest_district]

    if rest_category_main != 'All':
        data = data[data['combined_main_category_2']==rest_category_main]

    if rest_category != 'All':
        data = data[data['combined_main_category'].str.contains(rest_category)]

    return data.reset_index(drop=True)

def filter_data_scoring(data, rest_district, rest_category_main, rest_category):
    """
    Filters main dataframe based on district or restaurant selection
    FOR SCORING
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
        'user_ratings_total':'mean',
        'combined_rating':'mean'})\
        .rename(columns={'place_id':f'{rest_category.lower()}_restaurants'})

    return data.reset_index()

def merge_data(data, rest_district, rest_category_main, rest_category):
    """
    Creates a merged data set based on filtering selections and
    returns the final data set before normalization and scoring
    """
    if rest_category == 'All':
        data = filter_data_scoring(data, rest_district, rest_category_main, rest_category)
    else:
        data = filter_data_scoring(data, rest_district, 'All', 'All')\
            .merge(
                filter_data_scoring(data, rest_district, rest_category_main, rest_category)\
                    .drop(columns=['district','user_ratings_total','combined_rating']),
                how='left',
                on='district_cluster')\
            .fillna(0)
    return data

def score_data(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat):
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
    if rest_category == 'All':
        score_tot = score_com + score_pop + score_sat
        df_score['score'] = (score_com * (1-df_score['all_restaurants_norm'])\
                            +score_pop * df_score['user_ratings_total_norm']\
                            +score_sat * (1-df_score['combined_rating_norm']))\
                            / score_tot


    else:
        score_tot = 2 * score_com + score_pop + score_sat
        df_score['score'] = (score_com * (1-df_score['all_restaurants_norm'])\
                            +score_pop * df_score['user_ratings_total_norm']\
                            +score_sat * (1-df_score['combined_rating_norm'])
                            +score_com * (1-df_score[f'{rest_category.lower()}_restaurants_norm']))\
                            / score_tot

    # create output data_set
    df_output = pd.concat([df_merged, df_score], axis=1)\
        .merge(df_cluster_centers, how='left', on='district_cluster')
    return df_output

def pick_location(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat):
    """
    Select best / worst location based on custom scoring
    """
    if rest_district == 'All':
        n = 5
    else:
        n = 1

    best_location = score_data(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat)\
        .nlargest(n, 'score').reset_index(drop=True)
    worst_location = score_data(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat)\
        .nsmallest(n, 'score').reset_index(drop=True)

    return best_location, worst_location

def create_convexhull_polygon(map_object, list_of_points, layer_name, line_color, fill_color, weight, text):

    # Since it is pointless to draw a convex hull polygon around less than 3 points check len of input
    if len(list_of_points) > 2:

        # Create the convex hull using scipy.spatial
        form = [list_of_points[i] for i in ConvexHull(list_of_points).vertices]

        # Create feature group, add the polygon and add the feature group to the map
        fg = folium.FeatureGroup(name=layer_name)
        fg.add_child(folium.vector_layers.Polygon(locations=form, color=line_color, fill_color=fill_color,
                                                  weight=weight, stroke=False, popup=(folium.Popup(text))))
        map_object.add_child(fg)

    return (map_object)

#filter and search the restaurants - Let's test and remove - Burak
def search(df, category):
  search = lambda x:True if category.lower() in x.lower() else False
  venues = df[df['combined_main_category'].apply(search)].reset_index(drop='index')
  venues_lat_long = list(zip(venues['geometry.location.lat'], venues['geometry.location.lng']))
  return venues

# Functions END

# Required dictionary for restaurant dropdown menu
dict_rest = {
    'All':['All'],
    'European': ['All', 'French', 'Italian', 'Swiss', 'Portuguese', 'Spanish'],
    'Asian':['All', 'Japanese', 'Chinese', 'Thai', 'Indian', 'Other Asian'],
    'Middle Eastern & African': ['All', 'Lebanese', 'Turkish', 'Other Middle Eastern', 'African'],
    'American': ['All', 'American', 'South American', 'Mexican', 'Hawaiian'],
    'General': ['All', 'Restaurant', 'Bar / Pub / Bistro', 'Café'],
    'Fast Food':['All', 'Pizza', 'Hamburger', 'Chicken', 'Snacks / Take Away'],
    'Steakhouse / Barbecue / Grill': ['Steakhouse / Barbecue / Grill'],
    'Seafood': ['Seafood'],
    'Vegan / Vegetarian / Salad': ['Vegan / Vegetarian / Salad'],
    'All Other': ['All Other']}

# Required dictionary for area dropdown menu
list_district = [
    'All',
    'Bâtie - Acacias',
    'Champel',
    'Saint-Jean Charmilles',
    'Cité-Centre',
    'Eaux-Vives - Lac',
    'Grottes Saint-Gervais',
    'Jonction - Plainpalais',
    'La Cluse - Philosophes',
    'Pâquis Sécheron',
    'Servette Petit-Saconnex']

# Required dictionary for sliders
dict_slider1 = {'very low':0,
               'low':1,
               'neutral':2,
               'high':3,
               'very high':4}

dict_slider2 = {'very low':4,
               'low':3,
               'neutral':2,
               'high':1,
               'very high':0}


# Dropdown Menu START
st.sidebar.write('**Select Cuisine 🍽**')
rest_category_main = st.sidebar.selectbox("Main Restaurant Category", dict_rest.keys())
rest_category = st.sidebar.selectbox("Sub Restaurant Category", dict_rest[rest_category_main])
rest_district = st.sidebar.selectbox('Select Area 🗺',list_district)

st.sidebar.text("")
st.sidebar.write('**Select Scoring Criteria 🎯**')
#score_com = st.sidebar.slider('Number of Competitors', min_value=0, max_value=4, value=2, step=1)
score_com_slider = st.sidebar.select_slider('Number of Competitors', options=['very low', 'low', 'neutral', 'high', 'very high'], value='neutral')
score_pop_slider = st.sidebar.select_slider('Area Popularity', options=['very low', 'low', 'neutral', 'high', 'very high'], value='neutral')
score_sat_slider = st.sidebar.select_slider('Customer Satisfaction', options=['very low', 'low', 'neutral', 'high', 'very high'], value='neutral')

# Dropdown Menu END

score_pop = dict_slider1[score_pop_slider]
score_com = dict_slider2[score_com_slider]
score_sat = dict_slider2[score_sat_slider]

# filtered dataframe based on dropdpwn menu selection
df = filter_data(data, rest_district, rest_category_main, rest_category)

#create basic maps to be filled
lat = df_district[df_district['district']==rest_district]['district_lat']
lng = df_district[df_district['district']==rest_district]['district_lng']

# this can be part of the df later after finalization
zoom_start = {'All': 13.4,
            'Bâtie - Acacias': 15.4,
            'Champel': 14.4,
            'Cité-Centre': 15.4,
            'Eaux-Vives - Lac': 15,
            'Grottes Saint-Gervais': 15.4,
            'Jonction - Plainpalais': 15.4,
            'La Cluse - Philosophes': 15.4,
            'Pâquis Sécheron': 15.0,
            'Saint-Jean Charmilles': 15.0,
            'Servette Petit-Saconnex': 15.0}

geneva_1 = folium.Map(location=[lat, lng], zoom_start=zoom_start[rest_district], tiles='cartodbpositron')
geneva_2 = folium.Map(location=[lat, lng], zoom_start=zoom_start[rest_district], tiles='cartodbpositron')
geneva_3 = folium.Map(location=[lat, lng], zoom_start=zoom_start[rest_district], tiles='cartodbpositron')
geneva_4 = folium.Map(location=[lat, lng], zoom_start=zoom_start[rest_district], tiles='cartodbpositron')
geneva_5 = folium.Map(location=[lat, lng], zoom_start=zoom_start[rest_district], tiles='cartodbpositron')

# data['combined_main_category'].fillna(value='not defined', inplace=True) # not sure this is needed anymore - Burak

st.header('Next Restaurant in Geneva 👨🏻‍🍳🇨🇭')

# Map Section START

## Map 01 - Overview
marker_cluster = folium.plugins.MarkerCluster().add_to(geneva_1)

if len(df)!=0:
    for i,row in df.iterrows():
        folium.Marker(
            location=[row['geometry.location.lat'], row['geometry.location.lng']],
            popup=[row]
            ).add_to(marker_cluster)

# folium.LayerControl().add_to(geneva_1) # this gives a weird layer selection, so removed for the moment - Burak

## Map 02 - Price Levels
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

## Map 03 - Review Scores
group0 = folium.FeatureGroup(name='<span style=\\"color: red;\\">below 3.0</span>')
group1 = folium.FeatureGroup(name='<span style=\\"color: orange;\\">between 3.0 and 4.0</span>')
group2 = folium.FeatureGroup(name='<span style=\\"color: lightgreen;\\">between 4 and 4.5</span>')
group3 = folium.FeatureGroup(name='<span style=\\"color: green;\\">above 4.5</span>')
for i,row in df.iterrows():
    if row['combined_rating']<3.0:
        folium.CircleMarker(location=[row['geometry.location.lat'], row['geometry.location.lng']],radius=4, color='red',fillColor='red', fill=True,
                            popup=folium.Popup(f"{row['name']}, {row['combined_rating']}", max_width='100')).add_to(group0)
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

## Map 04 - Number of Reviews
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

## Map 05 - Best / Worst Location
best_locations = pick_location(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat)[0]
worst_locations = pick_location(data, rest_district, rest_category_main, rest_category, score_com, score_pop, score_sat)[1]

for i, row in best_locations.iterrows():
    cluster = row['district_cluster']
    list_of_points = data[data['district_cluster']==cluster][['geometry.location.lat','geometry.location.lng']].to_numpy()
    create_convexhull_polygon(geneva_5, list_of_points, layer_name='Best Locations',
                        line_color='green',
                        fill_color='green',
                        weight=1,
                        text=f"{row['district_cluster']}")

for i, row in worst_locations.iterrows():
    cluster = row['district_cluster']
    list_of_points = data[data['district_cluster']==cluster][['geometry.location.lat','geometry.location.lng']].to_numpy()
    create_convexhull_polygon(geneva_5, list_of_points, layer_name='Best Locations',
                        line_color='red',
                        fill_color='red',
                        weight=1,
                        text=f"{row['district_cluster']}")
    
if rest_category_main=='All' and rest_category=='All' and rest_district=='All':
    res = 'all restaurants in Geneva'
elif rest_category_main!='All' and rest_category=='All' and rest_district=='All':
    res = f'all {rest_category_main} restaurants in Geneva'
elif rest_category_main!='All' and rest_category!='All' and rest_district=='All':
    res = f'all {rest_category} restaurants in Geneva'
elif rest_category_main!='All' and rest_category!='All' and rest_district!='All':
    res = f'all {rest_category} restaurants in {rest_district}'
elif rest_category_main=='All' and rest_category=='All' and rest_district!='All':
    res = f'all restaurants in {rest_district}'
elif rest_category_main!='All' and rest_category=='All' and rest_district!='All':
    res = f'all {rest_category_main} restaurants in {rest_district}'

## Map Display
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺 Overview", "＄ Price Levels", "📊 Review Scores", "📈 Number of Reviews", "🟢🔴 Best/Worst Locations"])

with tab1:
    #st.header("## Check the best and worst restaurants based on general info")
    folium_static(geneva_1)
    st.write(f'The overview shows {res} 📍')
    st.write('Please use the dropdown menus on the left to make a selection')

with tab2:
    #st.markdown("## Check the best and worst restaurants based on Price Level")
    folium_static(geneva_2)
    st.write(f'The map shows the **Price Level** of {res} 📍')
    st.write(f'Use the checkboxes to filter your selection ☑️')

with tab3:
    #st.header("## Check the best and worst restaurants based on Review Score")
    folium_static(geneva_3)
    st.write(f'The map shows the **Review Score** of {res} 📍')
    st.write(f'Use the checkboxes to filter your selection ☑️')

with tab4:
    #st.header("## Check the best and worst restaurants based on Number of Reviews")
    folium_static(geneva_4)
    st.write(f'The map shows the **Number of Reviews** of {res} 📍')
    st.write(f'Use the checkboxes to filter your selection ☑️')

with tab5:
    #st.header("## Check the best and worst restaurants based on Number of Reviews")
    folium_static(geneva_5)
    st.write('The map shows the **Best Locations** in green and **Worst Locations** in red')
    st.write('Select the Criteria on the left to change the scoring')

# Map Section END

# districts = data['district'].unique() # is this required? - Burak

#geneva_zip_codes.fit_bounds([venues[['geometry.location.lat'][1]], venues['geometry.location.lng'][1]]) - is this required Burak
