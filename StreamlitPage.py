# UI
import streamlit as st
import streamlit.components.v1 as components

# SETUP
import requests
import json
import pickle
import os
from dotenv import load_dotenv
import traceback
import time

# DATA SCIENCE
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# GRAPH LIBRARIES
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# UTILS
from utils.SteamWrapper import get_Reviews
from utils.MakeNetwork import make_network

# NOTE THAT STEAM RETURNS PLAYTIME IN MINUTES
# https://partner.steamgames.com/doc/store/getreviews
load_dotenv()
steam_key = os.getenv('STEAM_KEY')

# SET UP THE PICKLE FILE FOR FUTURE READING
if not os.path.exists("game_id.pkl"):
    print("File does not exist")
    all_game_names = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/")
    json_names = json.loads(all_game_names.content)

    app_name_id = dict()
    for entry in json_names["applist"]["apps"]:
        app_name_id[entry['name'].lower()] = entry['appid']

    with open('game_id.pkl', 'wb') as fp:
        pickle.dump(app_name_id, fp)


# Function to create a sample plot
@st.cache_data
def create_plots(input_game_name, num_reviews):
    # INGEST THE USER INPUT AND FIND THE ID OF THE GAME
    input_game_name = input_game_name.lower()
    app_id_list = []
    with open('game_id.pkl', 'rb') as fp:
        app_id_dict = pickle.load(fp)
        try:
            # TODO FUZZY STRING MATCHING
            # LOOK FOR THE "PERFECT FIT" GAME
            if input_game_name in app_id_dict.keys():
                app_id_list.append(app_id_dict[input_game_name])
            # NO PERFECT FIT GAME
            else:
                app_id_list = [game_id for key, game_id in app_id_dict.items() if input_game_name in key]
        except KeyError as error:
            print("Error with finding the game name")

    app_id = app_id_list[0] if len(app_id_list) > 0 else 0

    reviews = get_Reviews(appid=app_id, reviewNum=num_reviews)
    game_info = reviews.pop(0)

    # GENERATE THE DATAFRAME THAT WILL BE USEFUL DOWN THE LINE
    review_author_id = []
    number_of_reviews = []
    review_recommendation = []  # 0 IS NOT RECOMMENDED, 1 IS RECOMMENDED
    review_playtime = []
    post_review_playtime = []
    review_total_playtime = []
    review_text = []
    for review in reviews:
        review_author_id.append(review['reviewAuthorSteamID'])
        number_of_reviews.append(review['numberOfReviews'])
        if review['reviewPositive']:
            review_recommendation.append("Yes")
        else:
            review_recommendation.append("No")
        playtime_review_temp = review['reviewPlaytimeAtReview'] / 60  # CONVERT MINUTES TO HOURS
        playtime_total_temp = review['reviewPlaytimeForever'] / 60  # CONVERT MINUTES TO HOURS
        review_playtime.append(playtime_review_temp)
        review_total_playtime.append(playtime_total_temp)
        post_review_playtime.append(playtime_total_temp - playtime_review_temp)
        review_text.append(review['reviewText'])

    review_data = {"AuthorID": review_author_id,
                   "num_reviews": number_of_reviews,
                   "recommended": review_recommendation,
                   "review_playtime": review_recommendation,
                   "continued_playtime": post_review_playtime,
                   "total_playtime": review_total_playtime,
                   "text": review_text}
    review_dataframe = pd.DataFrame(review_data)

    # CREATE THE PIECHART
    pie_chart = px.pie(review_dataframe, names='recommended', color='recommended',
                       title="Percent of reviews recommending", hover_data='recommended',
                       color_discrete_map={"Yes": 'darkblue', "No": 'darkred'})
    pie_chart.update_traces(hovertemplate=None)

    # PIE CHART OF THE NUMBER OF REVIEWS THAT AUTHORS LEFT
    bins = [0, 10, 50, 200, float('inf')]
    labels = ['0-10 Inexperienced Reviewer', '11-50 Experienced Reviewer', '51-200 Pro-Reviewer', '>200 Review Mogul']
    review_dataframe['binned'] = pd.DataFrame(
        pd.cut(review_dataframe['num_reviews'], bins=bins, labels=labels, include_lowest=True, right=False))
    number_of_reviews = px.pie(review_dataframe, names='binned', color='binned', labels=labels,
                               title="Experience of Reviewers", hover_data='binned')
    number_of_reviews.update_traces(hovertemplate=None)

    # SURVIVOR ANALYSIS REVIEWS
    average_survival = px.box(review_dataframe, y="recommended", x="total_playtime", color='recommended',
                              title="Playtime at review",
                              color_discrete_map={"Yes": 'darkblue', "No": 'darkred'})

    # SURVIVOR ANALYSIS POST REVIEWS
    post_completion_survival = px.box(review_dataframe, y="recommended", x="continued_playtime", color='recommended',
                                      title="Playtime post review",
                                      color_discrete_map={"Yes": 'darkblue', "No": 'darkred'})

    # MOVING AVERAGE TREND ANALYSIS
    positive_over_time = review_dataframe
    # WE CREATE BINS WITH THE NP ARRANGE, WITH EACH BIN CONTAINING 500 MINUTES
    largest_time_spent = positive_over_time['total_playtime'].max()
    if largest_time_spent > 500:
        largest_time_spent = 500

    bins = np.arange(0, largest_time_spent + 50, 50)
    positive_over_time['Time_Bin'] = pd.cut(positive_over_time['total_playtime'], bins, right=False)

    # WE FIND JUST THE NUMBER OF POSITIVE REVIEWS AND AGGREGATE INTO NEW COLUMNS THEM BASED ON SUM POS AND TOTAL COUNT
    # TODO LOG ODDS TO MINIMIZE THE VARIANCE
    positive_over_time['Positive_Review'] = review_dataframe["recommended"] == "Yes"
    grouped = positive_over_time.groupby('Time_Bin')['Positive_Review'].agg(
        [('Positive_Count', 'sum'), ('Total_Count', 'count')])
    grouped['Positive_Percentage'] = grouped['Positive_Count'] / grouped['Total_Count'] * 100

    # WE CREATE A WINDOW THAT WE CAN USE TO FIND THE MOVING AVERAGE
    window_size = 4  # Adjust the window size as needed
    grouped['MA_Positive_Percentage'] = grouped['Positive_Percentage'].rolling(window=window_size, min_periods=1).mean()

    line_chance_over_time = px.line(x=grouped.index.astype(str), y=grouped['MA_Positive_Percentage'],
                                    title="Recommendation odds over playtime")
    line_chance_over_time.update_traces(hovertemplate='Percent: %{y:.2f}%')
    line_chance_over_time.update_yaxes(ticksuffix="%")

    # TOPIC ANALYSIS
    vectorizer = TfidfVectorizer(stop_words='english', max_df=.97, min_df=.025)
    X = vectorizer.fit_transform(review_data['text'])

    nmf = NMF(n_components=10, init='random').fit(X)
    feature_names = vectorizer.get_feature_names_out()

    # FIND THE FIVE MOST IMPORTANT WORDS
    num_words_to_use = 5
    top_words = {}
    for topicID, filter in enumerate(nmf.components_):
        top_features_index = filter.argsort()[: -num_words_to_use - 1: -1]
        top_filter_words = [feature_names[i] for i in top_features_index]
        weights = filter[top_features_index]
        top_words[f"Topic {topicID}"] = [top_filter_words, weights]

    # MAKE THE SUBPLOTS WITH THE TOP WORDS
    nmf_topic_analysis = make_subplots(rows=2, cols=5, vertical_spacing=0.2)
    row = 1
    col = 1
    for key in top_words.keys():
        if col == 6:
            col = 1
            row += 1
        if (5 * (row - 1) + col) > 10:
            nmf_topic_analysis.add_trace(go.Bar(name=""), row, col)
        else:
            nmf_topic_analysis.add_trace(go.Bar(name=key, x=top_words[key][0], y=top_words[key][1]), row, col)
        col += 1

    nmf_topic_analysis.update_layout(showlegend=False, height=600)
    nmf_topic_analysis.update_traces(hovertemplate="<b>%{x}</b><extra></extra>")

    start_time = time.time()
    print(f"Processing begins at: {start_time}")

    users_games = []
    positive_review_user_libraries = []
    negative_review_user_libraries = []
    private_profiles = 0
    id_app_dict = {value: key for key, value in app_id_dict.items()}
    for index in review_dataframe.index:
        if index % 5 == 0:
            with loading_placeholder:
                st.write(f"Processed: {index} / {num_reviews}")

        user_owned_games = requests.get(
            f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={steam_key}&steamid={review_dataframe['AuthorID'][index]}&format=json")
        try:
            loaded_user_games = json.loads(user_owned_games.content)
        except Exception as e:
            private_profiles = private_profiles + 1
            continue
        if len(loaded_user_games["response"]) <= 0:
            private_profiles = private_profiles + 1
            continue
        top_games = []
        for game in loaded_user_games['response']['games']:
            if game['playtime_forever'] > 300:
                top_games.append([game['playtime_forever'], game['appid']])

        # THE NUMBER OF GAMES TO USE FOR THE CONNECTION GRAPH
        slice_index = 7
        sorted_array = sorted(top_games, key=lambda x: x[0], reverse=True)
        if len(sorted_array) < slice_index:
            slice_index = len(sorted_array)
        sorted_array = sorted_array[:slice_index]
        if len(sorted_array) == 0:
            private_profiles = private_profiles + 1
            continue

        game_id_list = [game_details[1] for game_details in sorted_array]
        game_name_list = []

        for ID in game_id_list:
            try:
                game_name_list.append(id_app_dict[ID])
            except KeyError as e:
                print("Key error")
        users_games.append(game_name_list)
        if review_dataframe['recommended'][index] == "Yes":
            positive_review_user_libraries.append(game_name_list)
        else:
            negative_review_user_libraries.append(game_name_list)

    end_time = time.time()
    print(f"Completed: {end_time}")
    print(f"Processing took {end_time - start_time}")

    # MAKE ALL THREE NETWORKS
    all_review_network = make_network(users_games)
    positive_review_network = make_network(positive_review_user_libraries)
    negative_review_network = make_network(negative_review_user_libraries)

    # # QUICK STAT FORMATS
    game_info["game_name"] = id_app_dict[app_id]
    game_info['averagePlaytimeFromReviews'] = review_dataframe['total_playtime'].mean()
    averaging_df = review_dataframe.groupby("recommended")
    # "averagePositivePlaytime"
    game_info['averagePositivePlaytime'] = averaging_df.get_group("Yes")['total_playtime'].mean()
    # "averageNegativePlaytime"
    game_info['averageNegativePlaytime'] = averaging_df.get_group("No")['total_playtime'].mean()
    game_info['private_profiles'] = private_profiles

    return game_info, pie_chart, number_of_reviews, average_survival, post_completion_survival, line_chance_over_time, nmf_topic_analysis, all_review_network, positive_review_network, negative_review_network


# Streamlit app layout and components
# LAYOUT
st.set_page_config(layout='wide')

# COMPONENTS
# HEADER AND TITLE
# st.image('./assets/SteamSense128.png')

# markdown_html = """
# <style>
# .bottom-aligned-header {
#     display: flex;
#     height: 200px;  /* Adjust height as needed */
#     align-items: end;
#     justify-content: center;
# }
# </style>
#
# <div class="bottom-aligned-header">
#     <h1 style="vertical-align:bottom">Title</h1>
# </div>
# """
# st.markdown(markdown_html, unsafe_allow_html=True)

st.title("GameSense: Steam Review Analytics Done Right")
st.markdown("By Tal Ashkenazi")
with st.container():
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        user_input = st.text_input("$$\Large \\text{Enter the name of the steam game: }$$", placeholder="Lunacid")
    with col2:
        num_reviews = st.number_input('$$\Large \\text{Number of reviews to query: }$$', min_value=50, format="%d", step=1, placeholder=50)
button_clicked = st.button('Analyze')

# Callback for button click
if button_clicked:
    captured_user_input = user_input
    # LOADING SCREEN TO INTERACT WITH USER DURING LOADING
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.subheader(f"Loading...")

    # GENERATE AND DISPLAY THE DIFFERENT TYPES OF GRAPHS
    try:
        quick_stats, pie1, pie2, average_survival, post_completion_survival, general_trend, topics, all_network, positive_net, negative_net = create_plots(
            captured_user_input,
            num_reviews)

        loading_placeholder.empty()
        with st.container():
            col1, col2, col3 = st.columns([0.3, 0.35, 0.35])
            with col1:
                st.header(f"Quick stats for {quick_stats['game_name']}")
                st.markdown(f"Overall score: :blue[**{quick_stats['reviewScoreDesc']}**]")
                st.markdown(f"Total reviews: :blue[ **{quick_stats['totalReviews']}**]")
                st.markdown(f"Total Positive Reviews: :blue[**{quick_stats['totalPositive']}**]")
                st.markdown(f"Total Negative Reviews: :blue[**{quick_stats['totalNegative']}**]")
                st.markdown(
                    'Average playtime: :blue[**{:.2f} hours**]'.format(quick_stats['averagePlaytimeFromReviews']))
                st.markdown(
                    'Average Positive Review playtime: :blue[**{:.2f} hours**]'.format(
                        quick_stats['averagePositivePlaytime']))
                st.markdown(
                    'Average Negative Review playtime: :blue[**{:.2f} hours**]'.format(
                        quick_stats['averageNegativePlaytime']))
            with col2:
                st.plotly_chart(pie1, use_container_width=True)
            with col3:
                st.plotly_chart(pie2, use_container_width=True)
        with st.container(border=True):
            st.header("Survival Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(average_survival)
                st.subheader("Example text")
                st.markdown("Explanation")
            with col2:
                st.plotly_chart(post_completion_survival)
                st.subheader("Example text")
                st.markdown("Explanation")
        with st.container(border=True):
            st.header("Playtime after review")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Example text**")
            with col2:
                st.plotly_chart(general_trend, use_container_width=True)
        with st.container(border=True):
            st.header("Topic analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(topics, use_container_width=True)
            with col2:
                st.markdown("**Example text**")
        st.header("User similarity analysis")
        tab1, tab2, tab3 = st.tabs(["All", "Positive", "Negative"])
        with tab1:
            st.plotly_chart(all_network, use_container_width=True)
        with tab2:
            st.plotly_chart(positive_net, use_container_width=True)
        with tab3:
            st.plotly_chart(negative_net, use_container_width=True)
        st.markdown(f"Excluded {quick_stats['private_profiles']} private profiles from data")
    except Exception as e:
        st.warning('Something went wrong, please try again', icon="⚠️")
        loading_placeholder.empty()
        print(traceback.print_exc())

# # EXAMPLE COMPONENTS
# st.warning('This is a warning', icon="⚠️")


# MOST PAIRWISE CONNECTIONS AND THEN RECOMMEND GAMES BASED ON THAT
