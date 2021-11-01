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
from datetime import datetime
from functools import reduce
import time
import toolz
import dask

st.set_page_config(layout="wide")

matplotlib.use("agg")

_lock = RendererAgg.lock

sns.set_style('darkgrid')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Historical Property Transactions by Postcode')


row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("This app examins transactions for properties by given postcode (either full or partial) using real time data from RightMove. The data only takes the latest transaction for each given property.")
    st.write("")
    st.markdown("More details can be accessed through Rightmove or by using the URL links attached in the Data Table (last section). The data is not always complete - it might miss the number of bedrooms/property type or number of bathrooms and this is purely a data defect. Only properties with the hyperlink were attached in the Data Table.")
    st.write("")
    st.write("")
    st.markdown(
        "**👈 To begin, please enter a valid postcode and start year**")
    st.write()


with st.sidebar:
    st.subheader("1. Postcode Filter")
    st.write("Please type a valid postcode and click Submit (it takes a few seconds to load).")
    
    user_input = st.sidebar.text_input("Input your postcode here")
    st.write("")
    user_input_year= st.sidebar.selectbox('Starting year:',
                                    [None, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
    st.write("*Starting year from which the latest transaction will be shown (until now)")
    st.write("")
    
    if user_input != '' and user_input_year != None:
        st.subheader("2. Choose Features")
        st.write("")
        st.write("Below you can choose the desired number of bedrooms and/or property type.")
        
        user_input_bedrooms = st.sidebar.selectbox('Choose the number of bedrooms:',
                                        [None, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        st.write("The Bedroom distribution bar chart shows count of bedrooms by property type.") 
        st.write("")
        user_input_property = st.sidebar.selectbox('Choose property type:',
                                        [None, 'Detached', 'Flat', 'Semi-Detached', 'Terraced'])
        st.write("The Property Type pie chart shows a proportion of property types.")
        st.write("")
        st.subheader("Hover over the plots to see details or to zoom in.")
        st.write("")

if user_input == '' or user_input_year == None:
    st.stop()
        
@st.cache(allow_output_mutation=True)
def get_data(user_input, user_input_year):
    
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
        if not items:
            return list()
        else:
            for i in range(len(items)):
            #    print(items)
                address.append(items[i].get("address"))
         #       if not items[i].get("address"):
         #           return list()
         #           break
                    
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
            
            #data = pd.DataFrame(data)
            #data['transactions_price'] = data.transactions_price.apply(lambda x: int(''.join(filter(str.isdigit, x))))
            #data['transactions_price'] = data['transactions_price'].apply(lambda x: "{:,}".format(x))
            
            return data
        
    #master = [dask.delayed(get_data_postcode)(i) for i in set(post_list_rightmove)]
    #df = dask.delayed(pd.concat)(master)
    #df = df.compute()
    
    master = [get_data_postcode(i) for i in set(post_list_rightmove)]
    master_filtered = [x for x in master if x]
    df = pd.DataFrame(reduce(lambda a, b: dict(a, **b), master_filtered))
    
    df['transactions_date_dt'] = df['transactions_date'].apply(lambda x: datetime.strptime(x, '%d %b %Y'))
    data_year = df[df['transactions_date_dt']>=str(user_input_year)+'-01-01 00:00:00']

    return data_year

if user_input != '':
    start_time = time.time()
    
    data_postcode = get_data(user_input, user_input_year)
    st.write("Dataset Loading Time", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
if data_postcode.empty:
    st.markdown("No property transactions recorded")
    st.stop()
    
        
line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

with line1_1:
    if len(data_postcode)>0:
        st.write("Data loaded")
    else:
        st.write("Invalid postcode")

    st.header('Analysis of properties traded since **{}** in **{}**'.format(user_input_year, user_input))
    st.write("")
    
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

@st.cache(suppress_st_warning=True)
def filter_data(data_filtered, user_input_bedrooms, user_input_property):
    
    if user_input_bedrooms != None:
        data_filtered = data_filtered[data_filtered['bedrooms'] == user_input_bedrooms]
        
    if user_input_property != None:
        data_filtered = data_filtered[data_filtered['propertyType'] == user_input_property]
        
    if data_filtered.empty:
        st.subheader('No propertis with given filter(s) found')
        
    else:
        data_filtered['lat_new'] = data_filtered['lat']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_filtered)) 
        data_filtered['lgt_new'] = data_filtered['lgt']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_filtered))
        
        fig = px.scatter_mapbox(data_filtered, lat="lat_new", lon="lgt_new", hover_name="address", color_discrete_sequence=["fuchsia"], zoom=12,
                                 width=1200)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
   
st.write('')
st.subheader("View Properties on Map")
st.write("Choose features on the left to filter by number of bedrooms and property type.")

data_postcode['transactions_price_numeric'] = data_postcode['transactions_price'].apply(lambda x: pd.to_numeric(re.sub('[^A-Za-z0-9]+', '',  x)))
#data_postcode['transactions_date_dt'] = data_postcode['transactions_date'].apply(lambda x: datetime.strptime(x, '%d %b %Y'))
#
#data_postcode_2020onwards = data_postcode[data_postcode['transactions_date_dt']>='2020-01-01 00:00:00']

mean_price = data_postcode['transactions_price_numeric'].mean()
no_properties = len(data_postcode['address'].unique())
md_results = f"The average price since the beginning of **{user_input_year}** for properties in this postcode is £**{round(mean_price,2):,}** with **{no_properties:,}** properties sold."
st.markdown(md_results)
st.write('')


if user_input_bedrooms != None and user_input_property != None:
    data_filtered = data_postcode
    data_filtered = data_filtered[data_filtered['bedrooms'] == user_input_bedrooms]
    data_filtered = data_filtered[data_filtered['propertyType'] == user_input_property]
    
    mean_price = data_filtered['transactions_price_numeric'].mean()
    no_properties = len(data_filtered['address'].unique())
    md_results = f"The average price since the beginning of **{user_input_year}** for the chosen number of bedrooms (**{user_input_bedrooms}**) and property type (**{user_input_property}**) in this postcode is £**{round(mean_price,2):,}** with **{no_properties:,}** properties sold."
    st.markdown(md_results)
    
if user_input_bedrooms != None and user_input_property == None:
    data_filtered = data_postcode
    data_filtered = data_filtered[data_filtered['bedrooms'] == user_input_bedrooms]
    mean_price = data_filtered['transactions_price_numeric'].mean()
    no_properties = len(data_filtered['address'].unique())
    md_results = f"The average price since the beginning of **{user_input_year}** for the chosen number of bedrooms (**{user_input_bedrooms}**) in this postcode is £**{round(mean_price,2):,}** with **{no_properties:,}** properties sold."
    st.markdown(md_results)
    
if user_input_bedrooms == None and user_input_property != None:
   data_filtered = data_postcode
   data_filtered = data_filtered[data_filtered['propertyType'] == user_input_property]
   mean_price = data_filtered['transactions_price_numeric'].mean()
   no_properties = len(data_filtered['address'].unique())
   md_results = f"The average price since the beginning of **{user_input_year}** for the chosen property type (**{user_input_property}**) in this postcode is £**{round(mean_price,2):,}** with **{no_properties:,}** properties sold."
   st.markdown(md_results)
    

if user_input_bedrooms == None and user_input_property == None:
    data_postcode['lat_new'] = data_postcode['lat']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_postcode)) 
    data_postcode['lgt_new'] = data_postcode['lgt']+ np.random.normal(loc=0.0, scale=0.00004, size=len(data_postcode))
    
    figD = px.scatter_mapbox(data_postcode, 
                             lat="lat_new", 
                             lon="lgt_new", 
                             hover_name="address", 
                             color_discrete_sequence=["fuchsia"], 
                             zoom=12,
                             width=1200)
    figD.update_layout(mapbox_style="open-street-map")
    figD.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) 
    st.plotly_chart(figD, use_container_width=True)
else:
    filter_data(data_postcode, user_input_bedrooms, user_input_property)
    
    

st.write("")
st.subheader("Data Table")
st.write("The below data shows all properties listed on RightMove. Choose features on the left to filter by number of bedrooms and property type.")

data_postcode['bedrooms'] = pd.to_numeric(data_postcode['bedrooms'], downcast='integer')
data_fil = data_postcode
    
table_loc = st.empty()

if user_input_bedrooms != None:
    data_fil = data_fil[data_fil['bedrooms']==user_input_bedrooms]

if user_input_property != None:
    data_fil = data_fil[data_fil['propertyType']==user_input_property]
    
data_fil = data_fil[['address', 'propertyType', 'bedrooms', 'bathrooms','transactions_price','transactions_date', 'transactions_tenure', 'detailUrl']]
table_loc.table(data_fil)

#data_fil = data_postcode[data_postcode['bedrooms']>=0]
#
#if (user_input_bedrooms == None and data_fil.empty):
#    st.write('There is no information of the chosen number of bedrooms - returning full dataset instead.')
#    data_fil = data_postcode
#    
#table_loc = st.empty()
#
#if user_input_bedrooms != None:
#    data_fil = data_fil[data_fil['bedrooms']==user_input_bedrooms]
#
#if user_input_property != None:
#    data_fil = data_fil[data_fil['propertyType']==user_input_property]
#    
#data_fil['bedrooms'] = pd.to_numeric(data_fil['bedrooms'], downcast='integer')
#data_fil = data_fil[['address', 'propertyType', 'bedrooms', 'bathrooms','transactions_price','transactions_date', 'transactions_tenure', 'detailUrl']]
#
#table_loc.table(data_fil)
#
#    st.markdown('***')
#    st.markdown(
#        "Thanks for going through this mini-analysis with me! I'd love feedback on this, so if you want to reach out you can find me on [twitter] (https://twitter.com/tylerjrichards) or my [website](http://www.tylerjrichards.com/).")
#
