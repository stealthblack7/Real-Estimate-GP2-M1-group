# main3.py

import streamlit as st
from streamlit_option_menu import option_menu
import subprocess
import os 
import pydeck as pd
import json
import pandas as pd
import numpy as np
import pydeck as pdk
from collections import deque
import openai
from PIL import Image
from datetime import datetime
import map as mp
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import accounte2 as acc
import folium
import geopandas
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
from PIL import Image


FACT_BACKGROUND = """
                    <div style="width: 100%;">
                        <div style="
                                    background-color: #ECECEC;
                                    border: 1px solid #ECECEC;
                                    padding: 1.5% 1% 1.5% 3.5%;
                                    border-radius: 10px;
                                    width: 100%;
                                    color: white;
                                    white-space: nowrap;
                                    ">
                          <p style="font-size:20px; color: black;">{}</p>
                        </div>
                    </div>
                    """     

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            back
            </style>
            <body style="background-color: aqua;">
    
    
            </body>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

this_x = 'Property Insight'
st.set_page_config("Property Insight", layout='centered')
with st.sidebar:
    page = option_menu(None, ["Home", "chat", "map",
                              "deals", "charts", "calc", "Account"],
                       icons=['house', 'chat-left-dots', 'geo-alt', 'archive', 'bar-chart-line', 'calculator', 'person'], menu_icon="cast", default_index=1)
    page

@st.cache_data
def map_data ():
    df = pd.read_excel("output/map.xlsx")
    return df

column_mapping = {
    'deal_nummper': 'رقم الصفقة',
    'real_estate_firts_text': 'المخطط',
    'real_estate_secando_text': 'القطعة',


}

def search_by_deal_nummper(deal_nummper):
    try:
        deal_nummper = int(deal_nummper)  # Convert user input to integer
    except ValueError:
        Invalid_input = "Invalid input. Please enter an integer for 'deal_nummper'."
        return Invalid_input

    # Filter the DataFrame based on the provided deal_nummper value
    filtered_df = df[df[column_mapping['deal_nummper']] == deal_nummper]

    if filtered_df.empty:
        st.write("No matching data found")
        
    else:
        # Display the matching data in the desired format
        for i, (index, row) in enumerate(filtered_df.iterrows(), 1):
            st.write(f"\nMatch {i}:")
            
            format_match(i, row)


def search_not_by_deal_nummper(real_estate_firts_text, real_estate_secando_text):
    # Filter the DataFrame based on the provided criteria
    filtered_df = df[
        (df[column_mapping['real_estate_firts_text']] == real_estate_firts_text) &
        (df[column_mapping['real_estate_secando_text']] == real_estate_secando_text)
    ]

    if filtered_df.empty:
        st.write("No matching data found.")
    else:
        # Display the matching data in the desired format with match numbers
        for i, (_, row) in enumerate(filtered_df.iterrows(), 1):
            w = i 
            st.write("------------------------------")
            st.write(f"\nMatch {w}:")
            format_match(i, row)


def search_by_real_estate_firts_text(real_estate_firts_text):
    # Filter the DataFrame based on the provided real_estate_firts_text
    filtered_df = df[df[column_mapping['real_estate_firts_text']]
                     == real_estate_firts_text]

    if filtered_df.empty:
        st.write("No matching data found")
    else:
        # Display all matching data in the desired format
        for i, (_, row) in enumerate(filtered_df.iterrows(), 1):
            st.write(f"\nMatch {i}:")
            format_match(i, row)
            


# Function to print a match in the desired format
def format_match(match_number, row):
    st.write(f"رقم الصفقة: {str(row['رقم الصفقة'])}\n")
    st.write(f"التصنيف: {str(row['التصنيف'])}\n")
    st.write(f"النوع: {str(row['النوع'])}\n")
    st.write(f"سعر الصفقة (ريال): {str(row['سعر الصفقة (ريال)'])}\n")
    st.write(f"المساحة (متر مربع): {str(row['المساحة (متر مربع)'])}\n")
    st.write(f"سعر المتر المربع (ريال) : {str(row['سعر المتر المربع (ريال)'])}\n")
    st.write(f"المنطقة: {str(row['المنطقة'])}\n")
    st.write(f"المدينة: {str(row['المدينة'])}\n")
    st.write(f"الحي: {str(row['الحي'])}\n")
    date_str = str(row['تاريخ الصفقة'])
    try:
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        formatted_date = date_obj.strftime("%d/%m/%Y")
        st.write(f"تاريخ الصفقة: {formatted_date}\n")
    except ValueError:
        # Print the date as is if it's not in the expected format
        st.write(f"تاريخ الصفقة: {date_str}\n")
    st.write(f"المخطط: {str(row['المخطط'])}\n")
    st.write(f"القطعة: {str(row['القطعة'])}\n\n")



if page == "Home":
    st.title("Welcome to the Home Page!")
    st.write("This is where your main content will go.")
    if st.button("Send"):
        for i in range(1, 6):
            st.write(f"\nMatch {i}:")

openai.api_key = "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"
chat_history = []
def get_response(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        stream=True

    )
    
    return response.choices[0].message


if page == "chat":
    st.title("Chat with ChatGPT")

    # Create a text input widget for user input
    user_input = st.text_input("You:", "")

    # Handle user input and display the conversation
    if st.button("Send"):
        user_message = user_input.strip()
        chat_history.append(f"You: {user_message}")
        response = get_response(user_message)
        chat_history.append(f"ChatGPT: {response}")

    # Display the chat history
    for message in chat_history:
        st.text(message)

@st.cache_data
def load_df():
    df = pd.read_excel(
        "output/map.xlsx", usecols=['القطعة', 'N', 'E', 'area', 'neighbor', 'المخطط', 'class', 'Type' , 'ID'])
    df.rename(columns={'N': 'Latitude', 'القطعة': 'Land', 'المخطط': 'plan',
              'E': 'Longitude'}, inplace=True)
    return df


def search_by_N_E(Latitude, Longitude):
    # Filter the DataFrame based on the provided criteria
    filtered_df = df[
        (df[column_mapping['Latitude']] == Latitude) &
        (df[column_mapping['Longitude']] == Longitude)
    ]

    if filtered_df.empty:
        st.write("No matching data found.")
    else:
        # Display the matching data in the desired format with match numbers
        for i, (_, row) in enumerate(filtered_df.iterrows(), 1):
            w = i
            st.write("------------------------------")
            st.write(f"\nMatch {w}:")
            format_match_map(i, row)

def format_match_map(match_number, row):
    st.write(f"التصنيف: {str(row['2التصنيف'])}\n")
    st.write(f"النوع: {str(row['2النوع'])}\n")
    st.write(f"المساحة (متر مربع): {str(row['area'])}\n")
    st.write(f"الحي: {str(row['الحي'])}\n")
    st.write(f"المخطط: {str(row['المخطط'])}\n")
    st.write(f"القطعة: {str(row['القطعة'])}\n\n")


def plot_from_df(df, folium_map):
    for i, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            tooltip=row['ID'],
            popup=f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}"
        ).add_to(folium_map)
    return folium_map


def init_map(center=(24.679708, 46.687859), zoom_start=10, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)


def create_point_map(df):
    # Cleaning
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].apply(
        pd.to_numeric, errors='coerce')
    # Convert PandasDataFrame to GeoDataFrame
    df['coordinates'] = df[['Latitude', 'Longitude']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = geopandas.GeoDataFrame(df, geometry='coordinates')
    df = df.dropna(subset=['Latitude', 'Longitude', 'coordinates'])
    return df

@st.cache_resource  # @st.cache_data
def load_map():
    # Load the map
    m = init_map()  # init
    df = load_df()  # load data
    m = plot_from_df(df, m)  # plot points
    return m


if page == "map":
    # Load the DataFrame for the "map" page
    df = load_df()  # Load the dataset

    m = load_map()
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None
    st.title('MAP')

    _, r1_col1, _ = st.columns([1, 10, 1])

    with r1_col1:
        level1_map_data = st_folium(m, height=520, width=600)
        st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']

        if st.session_state.selected_id is not None:
            try:
                # Convert to float and then to integer
                selected_id = int(float(st.session_state.selected_id))
                st.write(f'You Have Selected: {selected_id}')
                # st.write(f'DataFrame IDs: {df["ID"].values.tolist()}')
                # Check if the selected 'ID' exists in the DataFrame
                if selected_id in df['ID'].values:
                    # Retrieve the selected row from the dataset
                    selected_row = df[df['ID'] == selected_id].iloc[0]
                    # Display the entire selected row
                    st.write(selected_row)
                    # Display latitude and longitude
                    st.write(
                f"Latitude: {selected_row['Latitude']}, Longitude: {selected_row['Longitude']}")
                else:
                    st.write("No matching data found for the selected ID.")
            except ValueError:
                st.write("Invalid selected ID.")





        
@st.cache_data
def lode_data():
    df = pd.read_excel("output/modified_merged_data2.xlsx")
    return df 

if page == "deals":
    st.title("deals")
    with st.form("deal_nummper_user_inputs"):
        _ , r1_col1 , r1_col2 , r1_col3 , r1_col4 , _ = st.columns([1, 5, 2, 5,5, 1])
        _ , r2_col1 , r2_col2 , r2_col3 , r2_col4 , _ = st.columns([1, 5, 2, 5, 5, 1])
        with r1_col1:
            deal_nummper = st.text_input("Enter the deal number :" )
        with r1_col2:
            st.title("OR")
        with r1_col3 :
            real_estate_firts_numper = st.text_input(
                "  Enter the plan number :", )
        with r1_col4:
            real_estate_secando_numper = st.text_input(
                "Enter the land number :", )
        with r2_col1 :
            real_estate_numper_search_button = st.form_submit_button("search")
        

    if real_estate_numper_search_button:
        if deal_nummper !='' and real_estate_firts_numper == '' and real_estate_secando_numper == '':
            st.title("Search Results 1 ")
            df = lode_data()
            search_by_deal_nummper(int(deal_nummper))
            
        elif real_estate_secando_numper !='' and deal_nummper == '':
            input_data = {
            "real_estate_firts_text": real_estate_firts_numper,
            "real_estate_secando_text": real_estate_secando_numper,
            }
            st.title("Search Results2")
            df = pd.read_excel("output/modified_merged_data2.xlsx")
            search_not_by_deal_nummper(real_estate_firts_numper , real_estate_secando_numper)
        elif deal_nummper != '' and real_estate_firts_numper != '' and real_estate_secando_numper != '':
            st.warning(
                "Plaes enter the deal numper only or the plane number and the land number ", icon='🚨')
            # df = pd.read_excel("output/modified_merged_data2.xlsx")
            # search_by_real_estate_firts_text(real_estate_firts_numper)
        else:
            st.warning("ERROR", icon="🚨")


if page == "charts":
    _, r1_col1, r1_col2, r1_col3, r1_col4, r1_col5, _ = st.columns([2, 3, 3, 3, 3, 3, 2])
    _, r2_col1, r2_col2, r2_col3, r2_col4, r2_col5, _ = st.columns([2, 3, 3, 3, 3, 3, 2])
    _, r3_col1, _ = st.columns([6, 10, 1]) 
    _, r4_col1, _ = st.columns([1, 10, 1])
    with r1_col1:
        if st.button("2013"):
            with r3_col1:
                data_2013 = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2013')
                st.subheader('chart for 2013')
                with r4_col1:
                    st.bar_chart(data_2013, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data_2013, x='Neighborhood',
                                 y='Propertys Price  ',  height=350, use_container_width=True)
                

    with r1_col2:
        if st.button("2014"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2014')
                st.subheader('chart for 2014')
                with r4_col1 :
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r1_col3:
        if st.button("2015"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2015')
                st.subheader('chart for 2015')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r1_col4:
        if st.button("2016"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2016')
                st.subheader('chart for 2016')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)
    
    with r1_col5:
        if st.button("2017"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2017')
                st.subheader('chart for 2017')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r2_col1:
        if st.button("2018"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2018')
                st.subheader('chart for 2018')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r2_col2:
        if st.button("2019"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2019')
                st.subheader('chart for 2019')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)
    with r2_col3:
        if st.button("2020"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2020')
                st.subheader('chart for 2020')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                             y='PropertySelld ', height=350,use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r2_col4:
        if st.button("2021"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2021')
                st.subheader('chart for 2021')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)

    with r2_col5:
        if st.button("2022"):
            with r3_col1:
                data = pd.read_excel(
                    'output/2013.xlsx', sheet_name='2022')
                st.subheader('chart for 2022')
                with r4_col1:
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', height=350, use_container_width=True)
class_input_map = {
    'Commercial': 1,
    'Residential': 2,
}

type_input_map = {
    'Apartment': 1,
    'House': 2,
    'Plot of Land': 3,
    'Villa': 4,
}

neighbor_input_map = {
    'البيان': '01',
    'الرمال': '02',
    'الخير': '03',
    'العارض': '04',
    'القادسية': '05',
    'الملقا': '06',
    'المهديه': '07',
    'الياسمين': '08',
    'بنبان': '09',
    'طويق': '10',
    'لبن': '11',
}

if page == "calc":
    st.title("calc")
    _, r2_col1, _ = st.columns([1, 5, 1])

    with r2_col1:
        with st.form("user_inputs"):
            class1_input = st.selectbox(
                "Select Class:", ["Commercial", "Residential"])
            type1_input = st.selectbox(
                "Select Type:", ["Apartment", "House", "Plot of Land", "Villa"])
            area_input = st.number_input(
                "What is the area:", min_value=0, value=500)
            neighbor1_input = st.selectbox("Select Neighbor", [
                                           "البيان", "الرمال", "الخير", "العارض", "القادسية", "الملقا", "المهديه", "الياسمين", "بنبان", "طويق", "لبن"])
            date_input = st.date_input('Select a date')

            # Add the submit button
            submit_button = st.form_submit_button("Submit")

    class_input = class_input_map[class1_input]
    type_input = type_input_map[type1_input]
    neighbor_input = neighbor_input_map[neighbor1_input]

    # Extract year, month, and day from the date input
    year_input = date_input.year
    month_input = date_input.month
    day_input = date_input.day

    input_data = {
        'التصنيف2': int(class_input),
        'النوع2': int(type_input),
        'area': int(area_input),
        'الحي2': int(neighbor_input),
        'dd': int(day_input),
        'mm': int(month_input),
        'yyyy': int(year_input),
    }


    
# Now, submit_button is accessible globally
    if submit_button:
        # Convert input_data to a JSON-formatted string
        input_data_json = json.dumps(input_data)

    # Call ai6-6-22.py with the input data as an argument
        result = subprocess.run(
            ["python", "ai.py", input_data_json], capture_output=True, text=True)

        if result.returncode == 0:
            try:
                # Check if the output is not empty
                if result.stdout:
                    # Replace single quotes with double quotes and then parse as JSON
                    json_acceptable_string = result.stdout.replace("'", "\"")
                    predictions = json.loads(json_acceptable_string)

                # Display predictions
                    
                    _, r2_col1,  _ = st.columns(
                        [1, 5, 1])
                    with r2_col1:
                        st.title("Prediction")
                    # info sidebar
                        st.markdown(FACT_BACKGROUND.format(
                        predictions+"   \tSR"), unsafe_allow_html=True)
                    # st.text(predictions+" SR")
                else:
                    st.error("The AI script returned empty output.")
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON from the AI script: {str(e)}")
        else:
            # Handle any errors that occurred during script execution
            st.error("An error occurred while running the AI script.")


