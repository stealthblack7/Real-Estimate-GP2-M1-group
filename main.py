#  main.py

import streamlit as st
import streamlit_authenticator as stauth
from dependancies import sign_up, fetch_users
import dependancies as dep
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
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import folium
import geopandas
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
from PIL import Image
import openai
import requests
import datetime
import random
import time
from datetime import datetime




st.set_page_config(page_title='Property Insight',
                   initial_sidebar_state='collapsed')
# with st.sidebar:
#     page = option_menu(None, ["Home", "chat", "map",
#                               "deals", "charts", "calc", "Account"],
#                        icons=['house', 'chat-left-dots', 'geo-alt', 'archive', 'bar-chart-line', 'calculator', 'person'], menu_icon="cast", default_index=1)

page = option_menu(None, ["Home", "ChatBot", "Property Map",
                          "Deals", "Charts", "Prediction", "Account"],
                        icons=['house', 'chat-left-dots', 'geo-alt',
                               'archive', 'bar-chart-line', 'calculator', 'person'],
                        menu_icon="cast", default_index=0, orientation="horizontal")

#    header {visibility: hidden}

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
class_input_map = {
    'Commercial': 1,
    'Residential': 2,
}

type_input_map = {
    'Apartment': '01',
    'House': '02',
    'Plot of Land': '03',
    'Villa': '04',
}

neighbor_input_map = {
    'Albayan': '01',
    'Alrimal': '02',
    'Alkhayr': '03',
    'Alearid': '04',
    'Alqadisia': '05',
    'Almilqa': '06',
    'Almahdih': '07',
    'Alyasamin': '08',
    'Benban': '09',
    'Tuwaiq': '10',
    'Laban': '11',
}


def user_inputs():
    with st.form("user_inputs"):
        class1_input = st.selectbox(
            "Select Class:", ["Commercial", "Residential"])
        type1_input = st.selectbox(
            "Select Type:", ["Apartment", "House", "Plot of Land", "Villa"])
        area_input = st.number_input(
            "What is the area:", min_value=0, value=500)
        neighbor1_input = st.selectbox("Select Neighbor", [
            "Albayan", "Alrimal", "Alkhayr", "Alearid", "Alqadisia", "Almilqa", "Almahdih", "Alyasamin", "Benban", "Tuwaiq", "Laban"])
        date_input = st.date_input('Select a date')

        # Add the submit button
        submit_button = st.form_submit_button("Submit")
    return submit_button, class1_input, type1_input, area_input, neighbor1_input, date_input


def input_conv(class1_input, type1_input, area_input, neighbor1_input, date_input):
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
    return input_data


def predict(input_data, flag):
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
                if flag:
                    _, r1_col1, r1_col2, _ = st.columns([2, 5, 5, 1])
                    r1_col1.subheader("Prediction is :")
                    r1_col2.subheader(predictions+" SR")
                elif flag == False:
                    _, r1_col1, r1_col2, _ = st.columns([1, 5, 5, 1])
                    r1_col1.subheader("Prediction is :")
                    r1_col2.subheader(predictions+" SR")
                else:
                        st.error("The AI script returned empty output.")
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from the AI script: {str(e)}")

    else:
        # Handle any errors that occurred during script execution
        st.error(f"An error occurred while running the AI script: {result.stderr}")


if page == 'Account':
    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user['key'])
            usernames.append(user['username'])
            passwords.append(user['password'])

        credentials = {'usernames': {}}
        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {
                'name': emails[index], 'password': passwords[index]}

        Authenticator = stauth.Authenticate(
            credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)

        email, authentication_status, username = Authenticator.login(
            ':green[Login]', 'main')

        info, info1 = st.columns(2)

        if not authentication_status:
            sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    # let User see app
                    st.subheader('Accont')
                    st.markdown(
                        f"""
                    ---
                    Welcome {username} ❤️ 
                    """
                    )
                    Authenticator.logout('Log Out')
                elif not authentication_status:
                    with info:
                        st.error('Incorrect Password or username')
                else:
                    with info:
                        st.warning('Please feed in your credentials')
            else:
                with info:
                    st.warning('Username does not exist, Please Sign up')

    except:
        st.success('Refresh Page')


@st.cache_data
def map_data():
    df = pd.read_excel("data/map.xlsx")
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
            _, r1_col1, r1_col2, _ = st.columns([.1, 5, 5, 1])
            r1_col1.subheader(f"\nMatch {i}:")
            info = format_match(i, row)
            return info


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

            _, r1_col1, r1_col2, _ = st.columns([.1, 5, 5, 1])

            r1_col1.subheader(f"\nMatch {i}:")
            info = format_match(i, row)
            return info


class_map = {
    'تجاري': 'Commercial',
    'سكني': 'Residential',
}

type_map = {
    '1': 'Apartment',
    '2': 'House',
    '3': 'Plot of Land',
    '4': 'Villa',
    'شقة': 'Apartment',
    'بيت': 'House',
    'قطعة ارض': 'Plot of Land',
    'فيلا': 'Villa',
}

neighborhood_map = {
    'البيان': 'Albayan',
    'الرمال': 'Alrimal',
    'الخير': 'Alkhayr',
    'العارض': 'Alearid',
    'القادسية': 'Alqadisia',
    'الملقا': 'Almilqa',
    'المهدية': 'Almahdih',
    'الياسمين': 'Alyasamin',
    'بنبان': 'Benban',
    'طويق': 'Tuwaiq',
    'لبن': 'Laban',
}


def format_match(match_number, row):

    match_info = {
        "Deal Number": row["رقم الصفقة"],
        "class": class_map.get(row["التصنيف"], row["التصنيف"]),
        "Type": type_map.get(row["النوع"], row["النوع"]),
        "Deal Price (SR)": row["سعر الصفقة (ريال)"],
        "area": row["المساحة (متر مربع)"],
        "Price For one Square meters": row["سعر المتر المربع (ريال)"],
        "neighbor": neighborhood_map.get(row["الحي"], row["الحي"]),
        "Date": row["تاريخ الصفقة"],
        "plan": row["المخطط"],
        "Land": row["القطعة"]
    }
    return match_info


if page == "Home":
    st.title("Welcome to the Home Page!")
    st.markdown("""
                ---        
                """)
    st.write("This is where your main content will go.")
    if st.button("Send"):
        for i in range(1, 6):
            st.write(f"\nMatch {i}:")


openai.api_key = "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"

# Read website information from a file
with open("info.txt", "r") as info_file:
    website_info = info_file.read()

# Read website information from a file
with open("info.txt", "r") as info_file:
    website_info = info_file.read()


if page == "ChatBot":
    st.title("Chat with ChatGPT")
    st.markdown("""
                ---        
                """)

    # Initialize an empty conversation history
    conversation_history = []

    # Text input for user's message with a unique key
    user_message = st.text_input(
        "", key="user_message_input", placeholder="Send a message")

    if st.button("Send"):
        # Append the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_message})
        user_message_sptrip = user_message.strip()
        # botchat about the website 
        if user_message_sptrip.strip().lower() == "what do you do" or user_message_sptrip.strip().lower() == "talk about your web" or user_message_sptrip.strip().lower() == "what this web provide" :
            x = random.randint(1, 5)
            if x == 1:
                conversation_history.append(
                    {"role": "assistant", "content": "The Property Insight website offers users comprehensive information about real estate in Riyadh, KSA. You can explore properties on the map page, search for deals using our deal searcher, and utilize our AI and machine learning model on the predict page to estimate property prices. Additionally, we provide a market trends chart spanning the last 10 years. and I'm AI, I don't have feelings, but I'm here to help you with any questions or assistance you may need. How can I assist you today?"})
            if x == 2:
                conversation_history.append(
                    {"role": "assistant", "content": "Property Insight is a platform dedicated to providing users with valuable information about properties in Riyadh, KSA. Users can navigate properties on the map page, search for deals with the deal searcher, and leverage our AI and machine learning model to predict property prices on the predict page. Furthermore, we offer a market trends chart covering the past decade. and I'm AI, I don't have feelings, but I'm here to help you with any questions or assistance you may need. How can I assist you today?"})
            if x == 3:
                conversation_history.append(
                    {"role": "assistant", "content": "The Property Insight website is a valuable resource for individuals seeking property information in Riyadh, KSA. You can explore properties through the map page, search for deals with our deal searcher, and make use of our AI and machine learning model for property price predictions on the predict page. Additionally, we offer a market trends chart spanning the last 10 years. and I'm AI, I don't have feelings, but I'm here to help you with any questions or assistance you may need. How can I assist you today?"})
            if x == 4:
                conversation_history.append(
                    {"role": "assistant", "content": "Property Insight is your go-to website for accessing property-related information in Riyadh, KSA. You have the option to browse properties on the map page, search for deals with the deal searcher, and employ our AI and machine learning model for property price predictions on the predict page. We also present a market trends chart covering the past decade. and I'm AI, I don't have feelings, but I'm here to help you with any questions or assistance you may need. How can I assist you today?"})
            if x == 5:
                conversation_history.append(
                    {"role": "assistant", "content": "For those looking for property information in Riyadh, KSA, the Property Insight website is a valuable resource. You can explore properties on the map page, search for deals using the deal searcher, and utilize our AI and machine learning model for property price predictions on the predict page. In addition, we provide a market trends chart spanning the last 10 years. and I'm AI, I don't have feelings, but I'm here to help you with any questions or assistance you may need. How can I assist you today?"})
        # about the chatbot
        if user_message_sptrip.strip().lower() == "how are you" or user_message_sptrip.strip().lower() == "talk about your web" or user_message_sptrip.strip().lower() == "what this web provide":
            x = random.randint(1, 5)
            if x == 1:
                conversation_history.append(
                    {"role": "assistant", "content": "The Property Insight website offers users comprehensive information about real estate in Riyadh, KSA. You can explore properties on the map page, search for deals using our deal searcher, and utilize our AI and machine learning model on the predict page to estimate property prices. Additionally, we provide a market trends chart spanning the last 10 years."})

        else:
            try:
                # Generate a response from OpenAI
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history
                )

                # Extract and display the chatbot's reply
                chatbot_reply = response.choices[0].message["content"]

                # Append the chatbot's reply to the conversation history
                conversation_history.append(
                {"role": "assistant", "content": chatbot_reply})
                    

            except openai.error.RateLimitError as e:
                st.text("ChatBot: Rate limit exceeded. Waiting and retrying...")
                time.sleep(60)  # Wait for a minute before retrying

        # Display the entire chat history as a conversation
        for message in conversation_history:
            if message["role"] == "user":
                # Display user message
                st.markdown(
                    f'<div style="background-color: #2F4E75; color: white; padding: 10px; border-radius: 5px;"> You : {message["content"]}</div>',
                    unsafe_allow_html=True
                    )
            else:
                    st.markdown(
                    f'<div style="background-color: #3D6698; color: white; padding: 10px; border-radius: 5px;"> ChatBot : {message["content"]}</div>',
                    unsafe_allow_html=True
                    )


@st.cache_data
def load_df():
    df = pd.read_excel(
        "data/map.xlsx", usecols=['القطعة', 'N', 'E', 'area', 'neighbor', 'المخطط', 'class', 'Type', 'ID'])
    df.rename(columns={'N': 'Latitude', 'القطعة': 'Land', 'المخطط': 'plan',
              'E': 'Longitude'}, inplace=True)
    return df


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




if page == "Property Map":
    # Load the DataFrame for the "map" page
    df = load_df()  # Load the dataset
    _, r1_col1, _ = st.columns([1, 10, 1])
    m = load_map()
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None
    st.title('MAP')
    st.markdown("""
                ---        
                """)

    _, r1_col1, _ = st.columns([1, 10, 1])

    with r1_col1:
        level1_map_data = st_folium(m, height=520, width=600)
        st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']
        if st.session_state.selected_id is not None:
            try:
                # Convert to float and then to integer
                selected_id = int(float(st.session_state.selected_id))
        # Check if the selected 'ID' exists in the DataFrame
                if selected_id in df['ID'].values:
                    # Retrieve the selected row from the dataset
                    selected_row = df[df['ID'] == selected_id].iloc[0]
                    selected_row["neighbor"] = neighborhood_map.get(
                        selected_row["neighbor"], selected_row["neighbor"])

            # Display the entire selected row as a horizontal table
                    _, r1_col1,r1_col2, _ = st.columns([1, 5,5, 1])
                    _, r2_col1, _ = st.columns([1, 10, 1])
                    r1_col2.subheader('Area')
                    r1_col1.subheader('Neighbor')
                    r1_col2.write(str(selected_row['area']))
                    r1_col1.write(str(selected_row['neighbor']))
                    r1_col1.subheader('Land')
                    r1_col2.subheader('Plan')
                    r1_col1.write(str(selected_row['Land']))
                    r1_col2.write(str(selected_row['plan']))
                    r1_col1.subheader('Class')
                    r1_col2.subheader('Type')
                    r1_col1.write(str(selected_row['class']))
                    r1_col2.write(str(selected_row['Type']))
                    
                    # current_date = datetime.date.today()
                    current_date = datetime.now()

                    # Format the current date as "YYYY/MM/DD"
                    formatted_date = current_date.strftime("%Y/%m/%d")
                    with r2_col1 :
                        input_data = (input_conv(selected_row['class'], selected_row['Type'],
                                             selected_row['area'], selected_row['neighbor'], date_input=st.date_input("")))
                    predict(input_data , False)

                    
                else:
                    st.write("No matching data found for the selected ID.")
            except ValueError:
                st.write("Invalid selected ID.")


if page == "Deals":
    @st.cache_data
    def lode_data():
        df = pd.read_excel("data/modified_merged_data2.xlsx")
        return df

    df = load_df()
    
    st.title("Deals")
    st.markdown("""
                ---        
                """)
    with st.form("deal_nummper_user_inputs"):
        _, r1_col1, r1_col2, r1_col3, r1_col4, _ = st.columns(
            [1, 5, 2, 5, 5, 1])
        _, r2_col1, r2_col2, r2_col3, r2_col4, _ = st.columns(
            [1, 5, 2, 5, 5, 1])
        with r1_col1:
            deal_nummper = st.text_input("Enter the deal number :")
        with r1_col2:
            st.title("OR")
        with r1_col3:
            real_estate_firts_numper = st.text_input(
                "  Enter the plan number :", )
        with r1_col4:
            real_estate_secando_numper = st.text_input(
                "Enter the land number :", )
        with r2_col1:
            real_estate_numper_search_button = st.form_submit_button("search")

    _, r3_col1, _ = st.columns([0.1, 20, 0.1])
    with r3_col1:
        if real_estate_numper_search_button:
            if deal_nummper != '' and real_estate_firts_numper == '' and real_estate_secando_numper == '':
                
                st.title("Search Results ")
                df = lode_data()
                deal = search_by_deal_nummper(int(deal_nummper))
                _, r1_col1, r1_col2, _ = st.columns([.1, 5, 5, 1])
                r1_col2.subheader('Deal Number')
                r1_col1.subheader('Date')
                r1_col2.write(str(deal['Deal Number']))
                r1_col1.write(str(deal['Date']))
                r1_col2.subheader('Area')
                r1_col1.subheader('Neighbor')
                r1_col2.write(str(deal['area']))
                r1_col1.write(str(deal['neighbor']))
                r1_col1.subheader('Land')
                r1_col2.subheader('Plan')
                r1_col1.write(str(deal['Land']))
                r1_col2.write(str(deal['plan']))
                r1_col1.subheader('Class')
                r1_col2.subheader('Type')
                r1_col1.write(str(deal['class']))
                r1_col2.write(str(deal['Type']))
                r1_col1.subheader('Deal Price (SR)')
                r1_col2.subheader('Price/meter^2')
                r1_col1.write(str(deal['Deal Price (SR)']))
                r1_col2.write(str(deal['Price For one Square meters']))
            

            elif real_estate_secando_numper != '' and deal_nummper == '':
                input_data = {
                    "real_estate_firts_text": real_estate_firts_numper,
                    "real_estate_secando_text": real_estate_secando_numper,
                }
                st.title("Search Results ")
                df = lode_data()
                deal = search_not_by_deal_nummper(
                    real_estate_firts_numper, real_estate_secando_numper)
                st.table(deal)
            elif deal_nummper != '' and real_estate_firts_numper != '' and real_estate_secando_numper != '':
                st.warning(
                    "Plaes enter the deal numper only)(ex.5417889) or the plane number and the land number(ex.2566/ أ and 3783/2 ) ", icon='🚨')
            else:
                st.warning(
                    "Plaes enter the deal numper only)(ex.5417889) or the plane number and the land number(ex.2566/ أ and 3783/2 ) ", icon='🚨')


if page == "Charts":
    st.title("Charts")
    st.markdown("""
                ---        
                """)
    _, r1_col1, r1_col2, r1_col3, r1_col4, r1_col5, _ = st.columns(
        [2, 3, 3, 3, 3, 3, 2])
    _, r2_col1, r2_col2, r2_col3, r2_col4, r2_col5, _ = st.columns(
        [2, 3, 3, 3, 3, 3, 2])
    _, r3_col1, _ = st.columns([6, 10, 1])
    _, r4_col1, _ = st.columns([1, 10, 1])
    
    with r1_col1:
        if st.button("2013"):
            with r3_col1:
                data_2013 = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2013')
                st.subheader('chart for 2013')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data_2013, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data_2013, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)
    with r1_col2:
        if st.button("2014"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2014')
                st.subheader('chart for 2014')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r1_col3:
        if st.button("2015"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2015')
                st.subheader('chart for 2015')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r1_col4:
        if st.button("2016"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2016')
                st.subheader('chart for 2016')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r1_col5:
        if st.button("2017"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2017')
                st.subheader('chart for 2017')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)
    
    with r2_col1:
        if st.button("2018"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2018')
                st.subheader('chart for 2018')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r2_col2:
        if st.button("2019"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2019')
                st.subheader('chart for 2019')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)
    with r2_col3:
        if st.button("2020"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2020')
                st.subheader('chart for 2020')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r2_col4:
        if st.button("2021"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2021')
                st.subheader('chart for 2021')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)

    with r2_col5:
        if st.button("2022"):
            with r3_col1:
                data = pd.read_excel(
                    'data/2013.xlsx', sheet_name='2022')
                st.subheader('chart for 2022')
                with r4_col1:
                    st.markdown("""
                ---        
                """)
                    st.bar_chart(data, x='Neighborhood',
                                 y='PropertySelld ', color='#3D6698', height=350, use_container_width=True)
                    st.bar_chart(data, x='Neighborhood',
                                 y='Propertys Price  ', color='#3D6698', height=350, use_container_width=True)


if page == "Prediction":
    st.title("Prediction")
    st.markdown("""
                ---        
                """)
    _, r2_col1, _ = st.columns([1, 5, 1])
    
    with r2_col1:
        submit_button, class1_input, type1_input, area_input, neighbor1_input, date_input = user_inputs()
        input_data = input_conv(class1_input, type1_input,
                                area_input, neighbor1_input, date_input)
    if submit_button : 
        predict(input_data , True)

