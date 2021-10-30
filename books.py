from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
import xmltodict
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import gender_guesser.detector as gender
from streamlit_lottie import st_lottie
import requests
import base64
import requests
from bs4 import BeautifulSoup
import re
import json
from multiprocessing import Pool
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
import plotly.graph_objects as go
from dash import dash_table

st.set_page_config(layout="wide")

matplotlib.use("agg")

_lock = RendererAgg.lock

sns.set_style('darkgrid')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Historical Property Transactions by Postcode')


row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("This app will examin historical property prices by postcode - with real time data from RightMove")
    st.markdown(
        "**👈 To begin, please enter a valid postcode:**")
    st.write()


with st.sidebar:
    st.subheader("1. Postcode Filter")
    st.write("Please type a valid postcode and click Submit (it takes a few seconds to load).")
    user_input = st.sidebar.text_input(
        "Input your postcode here")
    
    if user_input:
        st.write("")
        st.write("The Property Type pie chart shows a proportion of property types.")
        st.write("")
        st.write("The Bedroom distribution bar chart shows count of bedrooms by property type.")
        st.write("")
        st.subheader("Hover over the plots to see details or to zoom in.")
        st.write("")
        st.subheader("2. Choose Features")
        st.write("")
        st.write("Below you can choose the desired number of bedrooms and/or property type. If you want to see everything just click Submit without filling the dropdowns.")

    else:
        st.markdown("No postcode typed")
        st.stop()
        
    user_input_bedrooms = st.sidebar.selectbox('Choose the number of bedrooms:',
                                    [None, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    st.write("")
    user_input_property = st.sidebar.selectbox('Choose property type:',
                                    [None, 'Detached', 'Flat', 'Semi-Detached', 'Terraced'])

@st.cache
def get_data(user_input):
    
    def urls(postcode):
        url_list = list()
        count = 1
        for i in range(1,41):
            url1 = 'https://www.rightmove.co.uk/house-prices/'
            url2 = '.html?page='
            a = url1 + postcode + url2 + str(count)
            url_list.append(a)
            count+=1
        return url_list
    
    post_list_rightmove = urls(user_input)
    
    def get_data_postcode(post_list):   
        address = list()
        propertyType = list()
        bedrooms = list()
        bathrooms = list()
        transactions_price = list()
        transactions_date = list()
        transactions_tenure = list()
        lat = list()
        lgt = list()
        detailUrl = list()
        
        def parse_html(html):
            page = requests.get(html)
            elem = BeautifulSoup(page.content, features="html.parser")
            results = elem.find('script',string=lambda text: 'location' in text.lower())
            text = ''
            for e in results.descendants:
                if isinstance(e, str):
                    text += e.strip()
                elif e.name in ['br',  'p', 'h1', 'h2', 'h3', 'h4','tr', 'th']:
                    text += '\n'
                elif e.name == 'li':
                    text += '\n- '
            return text
        
        results_text = parse_html(post_list)
        
        start = [m.start() for m in re.finditer('address', results_text)]
        end = start[1:]
        end.append(results_text.find('sidebar')-1)
        
        #Separate addresses
        items= list()
        for i in range(0,len(start)):
            substring = results_text[start[i]-2:end[i]-3]
            res = json.loads(substring)
            items.append(res)
        
        for i in range(len(items)):
        #    print(items)
            address.append(items[i].get("address"))
            propertyType.append(items[i].get("propertyType"))
            bedrooms.append(items[i].get("bedrooms"))
            bathrooms.append(items[i].get("bathrooms"))
            
            transaction = items[i].get("transactions")[0]
            transactions_price.append(transaction.get("displayPrice"))
            transactions_date.append(transaction.get("dateSold"))
            transactions_tenure.append(transaction.get("tenure"))
            
            loc = items[i].get("location")
            lat.append(loc.get("lat"))
            lgt.append(loc.get("lng"))
            detailUrl.append(items[i].get("detailUrl"))
                
        data = {'address': address, 
                'propertyType': propertyType, 
                'bedrooms':bedrooms, 
                'bathrooms':bathrooms,
                'transactions_price':transactions_price,
                'transactions_date':transactions_date,
                'transactions_tenure':transactions_tenure,
                'lat':lat,
                'lgt':lgt,
                'detailUrl':detailUrl}
        
        data = pd.DataFrame(data)
        data['transactions_price'] = data.transactions_price.apply(lambda x: int(''.join(filter(str.isdigit, x))))
        data['transactions_price'] = data['transactions_price'].apply(lambda x: "{:,}".format(x))
        
        return data
    
    data_master = pd.DataFrame(columns=['address','propertyType','bedrooms','bathrooms',
                                    'transactions_price','transactions_date',
                                     'transactions_tenure','lat','lgt','detailUrl'])
    
    data_master = [data_master.append(get_data_postcode(post_list_rightmove[i])) for i in range(len(post_list_rightmove))] 
    data_master = pd.concat(data_master)
    
    return data_master

if user_input != '':
    data_postcode = get_data(user_input)
        
line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

with line1_1:
    if len(data_postcode)>0:
        st.write("Data loaded")
    else:
        st.write("Invalid postcode")

    st.header('Analyzing historical prices for **{}**'.format(user_input))

    
has_records = any(data_postcode['bedrooms'])

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row3_1, _lock:
    st.subheader("Property Type")
    # plot the value
    df = pd.DataFrame(data_postcode[['address','propertyType']].groupby('propertyType').count()['address'])
    fig = px.pie(df, values='address', names=df.index)
    st.plotly_chart(fig, use_container_width=True)
        
with row3_2, _lock:
    st.subheader('Bedroom distribution')
    if has_records:
        fig_dist = px.histogram(data_postcode, x='bedrooms', color = 'propertyType')
        fig_dist.layout.showlegend = False
        fig_dist.update_layout(bargap=0.2)
        st.plotly_chart(fig_dist, use_container_width=True)  
    else:
        st.markdown(
            "We do not have information to find out the number of bedrooms")
        
st.write('')

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 1, .1, 1, .1))

@st.cache
def filter_data(data_filtered, user_input_bedrooms, user_input_property):
    
    if user_input_bedrooms != None:
        data_filtered = data_filtered[data_filtered['bedrooms'] == user_input_bedrooms]
        
    if user_input_property != None:
        data_filtered = data_filtered[data_filtered['propertyType'] == user_input_property]
    
    data_filtered['lat_new'] = data_filtered['lat']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_filtered)) 
    data_filtered['lgt_new'] = data_filtered['lgt']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_filtered))
    
    fig = px.scatter_mapbox(data_filtered, lat="lat_new", lon="lgt_new", hover_name="address", color_discrete_sequence=["fuchsia"], zoom=11,
                             width=1200)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig
   
st.write('')
st.subheader("View Properties on Map")
st.write("Choose features on the left to filter by number of bedrooms and property type.")
data_postcode['transactions_price_numeric'] = data_postcode['transactions_price'].apply(lambda x: pd.to_numeric(re.sub('[^A-Za-z0-9]+', '',  x)))
mean_price = data_postcode['transactions_price_numeric'].mean()
no_properties = len(data_postcode['address'].unique())
md_results = f"The average price for properties in this postcode is £**{mean_price:,}** and there are **{no_properties:,}** properties."
st.markdown(md_results)
    
if user_input_bedrooms == None and user_input_property == None:
    data_map = pd.DataFrame({'address': [user_input], 
            'lat': [data_postcode['lat'].median()], 
            'lgt':[data_postcode['lgt'].median()],
            'circle':[10]})
    
    figD = px.scatter_mapbox(data_postcode, 
                             lat="lat", 
                             lon="lgt", 
                             hover_name="address", 
                             color_discrete_sequence=["fuchsia"], 
                             zoom=12,
                             width=1200)
    figD.update_layout(mapbox_style="open-street-map")
    figD.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) 
    st.plotly_chart(figD, use_container_width=True)
else:
    figure_map = filter_data(data_postcode, user_input_bedrooms, user_input_property)
    st.plotly_chart(figure_map, use_container_width=True)
    

st.write("")
st.subheader("Data Table")
st.write("The below data shows all properties that contained a link for further reference. Choose features on the left to filter by number of bedrooms and property type.")


table_loc = st.empty()
data_fil = data_postcode[data_postcode['bedrooms']>=0]

if user_input_bedrooms != None:
    data_fil = data_fil[data_fil['bedrooms']==user_input_bedrooms]
    
if user_input_property != None:
    data_fil = data_fil[data_fil['propertyType']==user_input_property]
    
data_fil['bedrooms'] = pd.to_numeric(data_fil['bedrooms'], downcast='integer')
data_fil = data_fil[['address', 'propertyType', 'bedrooms', 'bathrooms','transactions_price','transactions_date', 'transactions_tenure', 'detailUrl']]

table_loc.table(data_fil)
#
#    st.markdown('***')
#    st.markdown(
#        "Thanks for going through this mini-analysis with me! I'd love feedback on this, so if you want to reach out you can find me on [twitter] (https://twitter.com/tylerjrichards) or my [website](http://www.tylerjrichards.com/).")
#
