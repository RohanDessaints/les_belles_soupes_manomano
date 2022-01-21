import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import geopandas as gpd
#import rtree
from shapely.geometry import Point
from shapely.ops import transform
import string
import re


#la base
image = Image.open('logo.png')
st.set_page_config(layout="wide")
st.markdown('Gaelle     Ghizlaine       Mbaye       Charles     Rohan')
st.image(image)

st.title("")
st.title("Improve customer retention")

st.markdown('Using ManoMano dataset about orders made on the website between August and November 2021,\
    we created a model that can predict if the order comes from a new or a regular customer.')
st.markdown('Thanks to machine learning, we have been able to isolate the new customers who could become regular ones\
    so that ManoMano team can orient specific actions toward this target or focus on the unsatisfaction points given in the comments.')
st.markdown('This dashboard will help ManoMano to transform first time customers into a regular customers')
unsafe_allow_html=True

st.title("")
#### Premier Graph ######



data = pd.read_csv('df_customers.csv')

mois = ['August','September','October','November','All']

# menu déroulant 'choix du mois'
menu = st.radio('Select a month',mois)
st.header("")

def metrics (data) :
    col1, col2, col3, col4, col5  = st.columns(5)
    col1.metric("Customers to retain", data.shape[0])
    col2.metric("Average bucket", str(round(data.bv_transaction.mean(),2))+'€')
    col3.metric("Comments", data.comment.notna().sum())
    col4.metric("Average Score", str(round(data.score.mean(),2))+'/10')
    col5.metric('Part of B2B', str(round(len(data[data['is_b2b']==True])/len(data)*100,2))+'%')


    st.title("")
    image2 = Image.open('cloud.png')
    st.image(image2)
    st.title("")


def graph1 (data) :
    st.title("")
    st.subheader('What do they order')
    fig2 = px.histogram(data, x="family", y="bv_transaction", histfunc='avg',
                  title = 'Product familys ordered by average bucket',
                 labels = {'family' : 'Product Family', 'bv_transaction':'Total Amount', 'month':'Month'})

    fig2.update_layout(bargap=0.1, title_x=0.5, yaxis_title= None, xaxis_title= None,)
    fig2.update_yaxes(title='Average bucket - €')
    fig2.update_xaxes(categoryorder='total descending')
    return st.plotly_chart(fig2)

def graph2 (data) :
    fig =  px.bar(data, x="category",
                 title = 'Top 10 product categories ordered',
                 labels = {'category' : 'Product Category', 'bv_transaction':'Total Amount', 'month':'Month'},
                 )

    fig.update_layout(
                    title_x=0.5,
                    yaxis_title= None,xaxis_title= None,
                    showlegend=False )
    fig.update_xaxes(categoryorder='total descending', range = [-0.5,10.5])
    return st.plotly_chart(fig)




# affichage des graphs
if menu == 'August' :
    august = data[data['month'] == 'August']
    metrics(august)
    graph1(august)
    graph2(august)
    #graph3(august)
    #graph4(august)

elif  menu == 'September' :
    september = data[data['month'] == 'September']
    metrics(september)
    graph1(september)
    graph2(september)
    #graph3(september)
    #graph4(september)

elif  menu == 'October' :
    october = data[data['month'] == 'October']
    metrics(october)
    graph1(october)
    graph2(october)
    #graph3(october)
    #graph4(october)

elif  menu == 'November' :
    november = data[data['month'] == 'November']
    metrics(november)
    graph1(november)
    graph2(november)
    #graph3(november)
    #graph4(november)

elif  menu == 'All' :
    metrics(data)
    graph1(data)
    graph2(data)
    #graph3(data)
    #graph4(data)

st.title("")
def graph3(data):
    st.subheader('When do they order')
    fig4 = px.sunburst(data_frame=data,
                       path=["day", 'day_time'],
                       color='day_time',
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       maxdepth=-1, branchvalues="total", template='seaborn', )

    fig4.update_traces(textinfo='label+percent entry', textfont_size=14)
    fig4.update_layout(margin=dict(t=20, l=20, r=20, b=20))
    fig4.update_layout(
        autosize=False,
        width=600,
        height=600, )
    #fig4.update_layout(title='DateTime Analysis')
    return st.plotly_chart(fig4)

graph3(data)
st.header("")
def graph4(data):
    st.subheader('How much do they order')
    df2 = round(data["bv_transaction_bucket"].value_counts() / len(data) * 100)
    fig = px.scatter(df2, size='value', color=df2, size_max=100, color_continuous_scale=px.colors.diverging.PiYG_r, )
    fig.update_yaxes(title='Percentage Orders')
    fig.update_xaxes(title='Transaction Bucket')
    #fig.update_layout(title='Percentage Orders per Transaction Bucket', title_x=0.5)

    return st.plotly_chart(fig)
graph4(data)

#### MAP #######
st.title("")
df_dpt = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson")
df_dpt['code'].replace('2A', '20A', inplace = True)
df_dpt['code'].replace('2B', '20B', inplace = True)
df_dpt = df_dpt.sort_values('code')
df_dpt.drop(df_dpt[df_dpt['code'] == '20B'].index, inplace = True)
df_dpt['code'].replace('20A', '2A', inplace = True)
df_dpt.reset_index(inplace = True)
df_dpt.drop(columns = 'index', inplace = True)

def zipcode_extract(x):
  if len(x) == 5:
    return x[0:2]
  else:
    return '0' + x[0]

# Ici, le fichier s'appelle 'table_globale.csv'. A adapter en fonction du nom de votre fichier.

df = pd.read_csv('table_globale.csv', sep = ',')
new_df = df[df['platform'] == 'FR']
new_df = new_df.drop(new_df[new_df['zipcode'].isna()].index)[['id', 'zipcode', 'score', 'first_order']]
new_df['zipcode'] = new_df['zipcode'].apply(lambda x : zipcode_extract(x))
new_df = new_df.groupby('zipcode').agg({'id' : 'count', 'score' : 'mean', 'first_order' : 'mean'}).reset_index()
new_df['zipcode'].replace('20', '2A', inplace = True)
new_df.drop(new_df[new_df['zipcode'] == '98'].index, inplace = True)

def center(x):
  for i in range(len(df_dpt)):
    if df_dpt.iloc[i,0] == x:
      return df_dpt.iloc[i,2].centroid
    else:
      continue

new_df['centroid'] = new_df['zipcode'].apply(lambda x : center(x))
new_df = gpd.GeoDataFrame(new_df, geometry = new_df['centroid'])
new_df.drop(columns = 'centroid', inplace = True)


def coord_center(point):
    c = []
    c.append(point.y)
    c.append(point.x)
    return c


new_df['coordinates'] = new_df['geometry'].apply(lambda x: coord_center(x))
new_df.rename(columns={'zipcode': 'code'}, inplace=True)


def icon(x):
  if new_df.iloc[i, 1] < 50:
    return 'lightblue'
  elif 50 <= new_df.iloc[i, 1] < 200:
    return 'darkblue'
  else:
    return 'black'

def rad(x):
  x = x/10
  return x
st.title("")

st.subheader("Where do they order")
st.header("")
map = folium.Map(location = [new_df['coordinates'][17][0], new_df['coordinates'][17][1]], zoom_start = 6)

map.choropleth(
    geo_data = df_dpt,
    name ='geometry',
    data = new_df,
    columns = ['code','first_order'],
    key_on = 'feature.properties.code',
    fill_color = 'Reds',
    fill_opacity = 0.5,
    line_opacity = 1.0,
    legend_name = 'Loyalty score'
)

folium.LayerControl().add_to(map)

for i in range(len(new_df)):
  location = new_df.iloc[i,5]
  libelle = f"Orders placed : {new_df.iloc[i, 1]}\nMean user xp score : {round(new_df.iloc[i, 2],2)}"
  marker = folium.Marker(tooltip = f'{df_dpt.iloc[i, 1]} ({df_dpt.iloc[i, 0]})', location = location, popup = libelle, icon=folium.Icon(icon = 'glyphicon glyphicon-euro', color = icon(i)))
  #marker = folium.vector_layers.CircleMarker(location = location, radius = rad(new_df.iloc[i, 1]), popup = libelle, tooltip = f'{df_dpt.iloc[i, 1]} ({df_dpt.iloc[i, 0]})')
  marker.add_to(map)
folium_static(map)

st.title("")

