# UI
# LINK FOR EVENTUAL DEPLOYMENT TO DOCKER CONTAINER
# https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
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

# TEXT PROCESSING
from thefuzz import fuzz, process

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
from utils.MakeNetwork import make_network, make_stacked_charts
from utils.TextCleaning import clean_text

# NOTE THAT STEAM RETURNS PLAYTIME IN MINUTES
# https://partner.steamgames.com/doc/store/getreviews
# ENVIRONMENT SETUP
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
def create_plots(input_game_name, num_reviews):
    # INGEST THE USER INPUT AND FIND THE ID OF THE GAME
    input_game_name = input_game_name.lower()
    app_id_list = []
    with open('game_id.pkl', 'rb') as fp:
        app_id_dict = pickle.load(fp)
        try:
            # FUZZY STRING MATCHING USING THEFUZZ
            match_ratios = process.extract(input_game_name, app_id_dict.keys(), scorer=fuzz.ratio)
            app_id_list.append(app_id_dict[match_ratios[0][0]])

        except Exception as error:
            st.warning("Error with finding the game name. Please check spelling and try again.", icon="⚠️")

    app_id = app_id_list[0] if len(app_id_list) > 0 else 0

    reviews = dict()
    try:
        reviews = get_Reviews(appid=app_id, reviewNum=num_reviews)
    except Exception as error:
        st.warning("Fatal error during API query. Please try again.", icon="⚠️")
        print(error)
        print(traceback.print_exc())
        return

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
        review_text.append(review['reviewText'].lower())

    review_data = {"AuthorID": review_author_id,
                   "num_reviews": number_of_reviews,
                   "recommended": review_recommendation,
                   "review_playtime": review_recommendation,
                   "continued_playtime": post_review_playtime,
                   "total_playtime": review_total_playtime,
                   "text": review_text}
    review_dataframe = pd.DataFrame(review_data)

    # CLEAN THE TEXT HERE
    with st.spinner('Cleaning review text...'):
        review_dataframe['text'] = review_dataframe['text'].apply(clean_text)

    # PIE CHART FOR SIMPLE PERCENT RECOMMENDED OPTIONS
    pie_chart = px.pie(review_dataframe, names='recommended', color='recommended',
                       title="Percent of reviews recommending", hover_data='recommended',
                       color_discrete_map={"Yes": '#2B66C2', "No": '#EB4339'})
    pie_chart.update_traces(hovertemplate=None)

    # PIE CHART OF THE NUMBER OF REVIEWS THAT AUTHORS LEFT
    bins = [0, 10, 50, 200, float('inf')]
    labels = ['0-10 Inexperienced Reviewer', '11-50 Experienced Reviewer', '51-200 Pro-Reviewer', '>200 Review Mogul']
    review_dataframe['binned'] = pd.DataFrame(
        pd.cut(review_dataframe['num_reviews'], bins=bins, labels=labels, include_lowest=True, right=False))
    # # THEORETICAL SUNBURST DIAGRAM, INNER CIRCLE IS TOO SMALL + LEGEND IS MISSING
    # number_of_reviews = px.sunburst(review_dataframe, path=['binned', 'recommended'],
    #                                 values=[1 for x in review_dataframe.index],
    #                                 color='binned',
    #                                 names='binned',
    #                                 labels=labels,
    #                                 title="Experience of Reviewers",
    #                                 hover_data=['binned'],
    #                                 branchvalues='total')
    number_of_reviews = px.pie(review_dataframe, names='binned', color='binned', labels=labels,
                               title="Experience of Reviewers", hover_data='binned')
    number_of_reviews.update_traces(hovertemplate=None)

    # SURVIVOR ANALYSIS REVIEWS
    average_survival = px.box(review_dataframe, y="recommended", x="total_playtime", color='recommended',
                              title="Playtime at review",
                              color_discrete_map={"Yes": 'darkblue', "No": 'darkred'})
    average_survival.update_xaxes(title="Playtime at time of review (hours)")

    # SURVIVOR ANALYSIS POST REVIEWS
    post_completion_survival = px.box(review_dataframe, y="recommended", x="continued_playtime", color='recommended',
                                      title="Playtime post review",
                                      color_discrete_map={"Yes": 'darkblue', "No": 'darkred'})
    post_completion_survival.update_xaxes(title="Playtime post review (hours)")

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
    line_chance_over_time.update_xaxes(title="Total Playtime (hours)")
    line_chance_over_time.update_yaxes(title="Percent recommended", ticksuffix="%")

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

    # TODO: MORE TEXT PLOTTED AGAINST PERCENT TO RECOMMEND
    # TODO: MAYBE? SUMMARIZATION OF THE TEXT USING SOME SUMMARIZATION EXTRACTION, THEN FEED IT TO CHAT GPT TO MAKE IT MORE READABLE

    start_time = time.time()
    print(f"Processing begins at: {start_time}")

    users_games = []
    positive_review_user_libraries = []
    negative_review_user_libraries = []
    private_profiles = 0
    id_app_dict = {value: key for key, value in app_id_dict.items()}

    start_time_estimate = time.time()
    for index in review_dataframe.index:
        # WHAT PERCENT OF THE TASK IS COMPLETE
        progress = (index + 1) / num_reviews
        elapsed_time = time.time() - start_time_estimate
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time

        # Convert remaining time to minutes
        remaining_minutes = int(remaining_time // 60)
        remaining_seconds = int(remaining_time % 60)
        loading_placeholder.progress(progress,
                                     text=f"Processed: {index} / {num_reviews}. Estimated time remaining: {remaining_minutes} min {remaining_seconds} sec")

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

    # MAKE THE HORIZONTAL STACKED BAR CHART FOR EACH OF THE THREE NETWORKS
    all_stacked_chart = make_stacked_charts(users_games)
    positive_stacked_chart = make_stacked_charts(positive_review_user_libraries)
    negative_stacked_chart = make_stacked_charts(negative_review_user_libraries)

    # # QUICK STAT FORMATS
    game_info["game_name"] = id_app_dict[app_id]
    game_info['averagePlaytimeFromReviews'] = review_dataframe['total_playtime'].mean()
    averaging_df = review_dataframe.groupby("recommended")
    # "averagePositivePlaytime"
    game_info['averagePositivePlaytime'] = averaging_df.get_group("Yes")['total_playtime'].mean()
    # "averageNegativePlaytime"
    game_info['averageNegativePlaytime'] = averaging_df.get_group("No")['total_playtime'].mean()
    game_info['private_profiles'] = private_profiles

    return game_info, pie_chart, number_of_reviews, average_survival, post_completion_survival, line_chance_over_time, nmf_topic_analysis, [
        all_review_network, positive_review_network, negative_review_network], [all_stacked_chart,
                                                                                positive_stacked_chart,
                                                                                negative_stacked_chart]


# Streamlit app layout and components
# LAYOUT
st.set_page_config(page_title="Steam Sense", page_icon='./assets/SteamSense128.ico', layout='wide')

# COMPONENTS
# HEADER AND TITLE
st.title("Steam Sense: Steam Review Analytics Done Right")
st.markdown("By Tal Ashkenazi - [Github](https://github.com/tal-ashkenazi01)")
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        user_input = st.text_input("$$\Large \\text{Enter the name of the steam game: }$$", value="Lunacid")
    with col2:
        num_reviews = st.number_input('$$\Large \\text{Select Number of Reviews: }$$', min_value=50, format="%d",
                                      step=1, placeholder=50,
                                      help="Going above 5000 reviews not recommended and will lead to long wait times")
button_clicked = st.button('Analyze')

# Callback for button click
if button_clicked:
    captured_user_input = user_input
    # LOADING SCREEN TO INTERACT WITH USER DURING LOADING
    loading_placeholder = st.progress(0, text=None)

    # GENERATE AND DISPLAY THE DIFFERENT TYPES OF GRAPHS
    try:
        quick_stats, pie1, pie2, average_survival, post_completion_survival, general_trend, topics, all_networks, all_percentages = create_plots(
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
                st.subheader("How Many Hours Did Reviewers Play?")
                st.markdown("Explanation")
            with col2:
                st.plotly_chart(post_completion_survival)
                st.subheader("How Many Hours Were Played Post-Review?")
                st.markdown("Explanation")
        with st.container():
            st.header("Likelihood of Recommending Game Over Playtime")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Example text**")
            with col2:
                st.markdown("$$\large \\text{Compare recommendations over playtime to large titles: }$$")
                trend1, trend2, trend3, trend4, trend5, trend6, trend7, trend8 = st.tabs(
                    [quick_stats['game_name'], "Elden Ring", "Counter-Strike 2", "Starfield", "No Man's Sky",
                     "Terraria", "NBA 2K24", "Apex Legends"])
                with trend1:
                    st.plotly_chart(general_trend, use_container_width=True)
                with trend2:
                    with open('./assets/EldenRingExample.json', 'r') as file:
                        Elden_Ring_chart_data = json.load(file)
                    ER_fig = go.Figure(data=Elden_Ring_chart_data['data'], layout=Elden_Ring_chart_data['layout'])
                    st.plotly_chart(ER_fig, use_container_width=True)
                with trend3:
                    with open('./assets/CS2Example.json', 'r') as file:
                        CS2_chart_data = json.load(file)
                    CS2_fig = go.Figure(data=CS2_chart_data['data'], layout=CS2_chart_data['layout'])
                    st.plotly_chart(CS2_fig, use_container_width=True)
                with trend4:
                    with open('./assets/StarfieldExample.json', 'r') as file:
                        Starfield_chart_data = json.load(file)
                    S_fig = go.Figure(data=Starfield_chart_data['data'], layout=Starfield_chart_data['layout'])
                    st.plotly_chart(S_fig, use_container_width=True)
                with trend5:
                    with open('./assets/NoManSkyExample.json', 'r') as file:
                        NoMansSky_chart_data = json.load(file)
                    NMS_fig = go.Figure(data=NoMansSky_chart_data['data'], layout=NoMansSky_chart_data['layout'])
                    st.plotly_chart(NMS_fig, use_container_width=True)
                with trend6:
                    with open('./assets/TerrariaExample.json', 'r') as file:
                        Terraria_chart_data = json.load(file)
                    Terraria_fig = go.Figure(data=Terraria_chart_data['data'], layout=Terraria_chart_data['layout'])
                    st.plotly_chart(Terraria_fig, use_container_width=True)
                with trend7:
                    with open('./assets/NBA2K24Example.json', 'r') as file:
                        NBA_chart_data = json.load(file)
                    NBA_fig = go.Figure(data=NBA_chart_data['data'], layout=NBA_chart_data['layout'])
                    st.plotly_chart(NBA_fig, use_container_width=True)
                with trend8:
                    with open('./assets/ApexLegendsExample.json', 'r') as file:
                        Apex_chart_data = json.load(file)
                    Apex_fig = go.Figure(data=Apex_chart_data['data'], layout=Apex_chart_data['layout'])
                    st.plotly_chart(Apex_fig, use_container_width=True)
        with st.container(border=True):
            st.header("Topic analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(topics, use_container_width=True)
            with col2:
                st.markdown("**Example text**")
        st.header("User similarity analysis")
        st.markdown("**explanation goes here**")
        st.markdown(f":red[Excluded {quick_stats['private_profiles']} private profiles from data]")
        tab1, tab2, tab3 = st.tabs(["All", "Positive", "Negative"])
        # TODO EACH GAME AS A PERCENT OF THE TOP GAMES OF PLAYERS
        with tab1:
            st.header("Most Played Games by Reviewers")
            st.plotly_chart(all_networks[0], use_container_width=True)
            st.plotly_chart(all_percentages[0], use_container_width=True)
        with tab2:
            st.header("Most Played Games by Fans")
            st.plotly_chart(all_networks[1], use_container_width=True)
            st.plotly_chart(all_percentages[1], use_container_width=True)
        with tab3:
            st.header("Most Played Games by Critics")
            st.plotly_chart(all_networks[2], use_container_width=True)
            st.plotly_chart(all_percentages[2], use_container_width=True)
    except Exception as e:
        st.warning('Something went wrong, please try again', icon="⚠️")
        loading_placeholder.empty()
        print(traceback.print_exc())

# # EXAMPLE COMPONENTS
# st.warning('This is a warning', icon="⚠️")


# MOST PAIRWISE CONNECTIONS AND THEN RECOMMEND GAMES BASED ON THAT
