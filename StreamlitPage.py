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
import random

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
from utils.SteamWrapper import get_Reviews, get_gameplay_video
from utils.MakeNetwork import make_network, make_stacked_charts
from utils.TextCleaning import clean_text, cohen_d, u_test, generate_topics

# NOTE THAT STEAM RETURNS PLAYTIME IN MINUTES
# https://partner.steamgames.com/doc/store/getreviews
# ENVIRONMENT SETUP
load_dotenv()
steam_key = os.getenv('STEAM_KEY')
GOOGLE_API_TOKEN = os.getenv('GOOGLE_API_KEY')
HUGGING_API_TOKEN = os.getenv('HUGGING_API_KEY')

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
            st.Error("Error with finding the game name. Please check spelling and try again.", icon="🚨️")

    app_id = app_id_list[0] if len(app_id_list) > 0 else 0

    try:
        with st.spinner('Fetching reviews...'):
            reviews = get_Reviews(appid=app_id, reviewNum=num_reviews)
    except Exception as error:
        st.Error("Fatal error during API query. Please try again.", icon="🚨️")
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
                   "review_playtime": review_playtime,
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

    # CREATE THE EXPERIENCE PIE CHART
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
    vectorizer = TfidfVectorizer(max_df=.95, min_df=.025, stop_words='english')
    X = vectorizer.fit_transform(review_dataframe['text'])

    nmf = NMF(n_components=10, init='random').fit(X)
    feature_names = vectorizer.get_feature_names_out()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # FIND THE FIVE MOST IMPORTANT WORDS
    num_words_to_use = 5
    top_words = {}
    for topicID, filter in enumerate(nmf.components_):
        top_features_index = filter.argsort()[: -num_words_to_use - 1: -1]
        top_filter_words = [feature_names[i] for i in top_features_index]
        weights = filter[top_features_index]
        scaled_weights = scaler.fit_transform(np.array(weights).reshape(-1, 1))  # CHANGE TO A 2D ARRAY
        top_words[f"Topic {topicID}"] = [top_filter_words, scaled_weights.flatten()]

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

    # TODO: KL DIVERGENCE
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

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
    game_info["app_id"] = app_id
    game_info['averagePlaytimeFromReviews'] = review_dataframe['total_playtime'].mean()
    game_info['averagePlaytimeOnReview'] = review_dataframe['review_playtime'].mean()
    game_info['medianPlaytimeOnReview'] = review_dataframe['review_playtime'].median()
    game_info['averagePlaytimePostReview'] = review_dataframe['continued_playtime'].mean()
    game_info['medianPlaytimePostReview'] = review_dataframe['continued_playtime'].median()

    averaging_df = review_dataframe.groupby("recommended")
    # POSITIVE REVIEW STATS
    #######################
    # POSITIVE REVIEW FLAG
    positive_review_flag = True if "Yes" in averaging_df['recommended'].unique() else False

    # TOTAL
    if positive_review_flag:
        game_info['averagePositiveTotalPlaytime'] = averaging_df.get_group("Yes")['total_playtime'].mean()

        # ON REVIEW
        game_info['averagePositivePlaytime'] = averaging_df.get_group("Yes")['review_playtime'].mean()
        game_info['medianPositivePlaytime'] = averaging_df.get_group("Yes")['review_playtime'].median()

        # POST REVIEW
        game_info['averagePositiveContinuedPlaytime'] = averaging_df.get_group("Yes")['continued_playtime'].mean()
        game_info['medianPositiveContinuedPlaytime'] = averaging_df.get_group("Yes")['continued_playtime'].median()
    else:
        data_info.warning("No positive reviews found. Consider increasing sample size.", "⚠️")
        game_info['averagePositiveTotalPlaytime'] = 0
        game_info['averagePositivePlaytime'] = 0
        game_info['medianPositivePlaytime'] = 0
        game_info['averagePositiveContinuedPlaytime'] = 0
        game_info['medianPositiveContinuedPlaytime'] = 0

    # NEGATIVE REVIEW STATS
    #######################
    # FLAG IF THERE ARE NO NEGATIVE REVIEWS:
    negative_review_flag = True if "No" in averaging_df['recommended'].unique() else False

    if negative_review_flag:
        # TOTAL
        game_info['averageNegativeTotalPlaytime'] = averaging_df.get_group("No")['total_playtime'].mean()

        # ON REVIEW
        game_info['averageNegativePlaytime'] = averaging_df.get_group("No")['review_playtime'].mean()
        game_info['medianNegativePlaytime'] = averaging_df.get_group("No")['review_playtime'].median()

        # POST-REVIEW
        game_info['averageNegativeContinuedPlaytime'] = averaging_df.get_group("No")['continued_playtime'].mean()
        game_info['medianNegativeContinuedPlaytime'] = averaging_df.get_group("No")['continued_playtime'].median()
    else:
        data_info.warning("No negative reviews found. Consider increasing sample size.", icon="⚠️")
        game_info['averageNegativeTotalPlaytime'] = 0
        game_info['averageNegativePlaytime'] = 0
        game_info['medianNegativePlaytime'] = 0
        game_info['averageNegativeContinuedPlaytime'] = 0
        game_info['medianNegativeContinuedPlaytime'] = 0

    if positive_review_flag and negative_review_flag:
        # CALCULATE THE MAGNINITUDE OF DIFFERENCES WITH COHEN'S D
        game_info["cohens_d"] = (
            cohen_d(averaging_df.get_group("Yes")['review_playtime'], averaging_df.get_group("No")['review_playtime']))
        game_info["r_statistic_at_review"] = u_test(averaging_df.get_group("Yes")['review_playtime'],
                                                    averaging_df.get_group("No")['review_playtime'])
        game_info["r_statistic_post_review"] = u_test(averaging_df.get_group("Yes")['continued_playtime'],
                                                      averaging_df.get_group("No")['continued_playtime'])
    else:
        game_info["cohens_d"] = 0
        game_info["r_statistic_at_review"] = 0
        game_info["r_statistic_post_review"] = 0

    # GET TOPWORDS:
    game_info["top_words"] = [x[0] for x in top_words.values()]

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
alert_placeholder = st.empty()
st.title("Steam Sense: Steam Review Analytics")
st.markdown("By Tal Ashkenazi - [Github](https://github.com/tal-ashkenazi01)")
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        text_input_placeholder = st.empty()
        user_input = text_input_placeholder.text_input("$$\Large \\text{Enter the name of the steam game: }$$",
                                                       value="Lunacid")
    with col2:
        number_input_placeholder = st.empty()
        num_reviews = number_input_placeholder.number_input('$$\Large \\text{Select Number of Reviews: }$$',
                                                            min_value=50, format="%d",
                                                            step=1, value=50,
                                                            help="Going above 5000 reviews not recommended and will lead to long wait times")
    button_centering_col = st.columns([0.25, 0.25, 0.25, 0.25])
    with button_centering_col[1]:
        button_clicked = st.button('Analyze', use_container_width=True,
                                   help="Input any title from the Steam store above, then click analyze!")
    with button_centering_col[2]:
        random_game = st.button("Don't know any games? Click me!", use_container_width=True,
                                help="Choose a random game to analyze. This still uses your number of input reviews")
    data_info = st.empty()

# Callback for button click
if button_clicked or random_game:
    captured_user_input = user_input
    if num_reviews < 100:
        alert_placeholder.warning(
            "Warning: Running analysis with less than 100 reviews will lead to low quality results", icon="⚠️")

    if random_game:
        captured_user_input = random.choice(
            ["Inscryption", "NBA 2K23", "Slay the Spire", "Tom Clancy's Rainbow Six Siege", "PUBG: Battlegrounds",
             "Sid Meier’s Civilization vi", "Baldur's Gate 3", "Counter-Strike 2", "EA SPORTS FC 24"])
        text_input_placeholder.text_input("$$\Large \\text{Enter the name of the steam game: }$$",
                                          value=captured_user_input)

    # LOADING SCREEN TO INTERACT WITH USER DURING LOADING
    loading_placeholder = st.progress(0, text=None)

    # GENERATE AND DISPLAY THE DIFFERENT TYPES OF GRAPHS
    try:
        quick_stats, pie1, pie2, average_survival, post_completion_survival, general_trend, topics, all_networks, all_percentages = create_plots(
            captured_user_input,
            num_reviews)

        loading_placeholder.empty()
        with st.container():
            title_text_col = st.columns([0.61, 0.39])
            with title_text_col[0]:
                st.markdown(f"### Results for {quick_stats['game_name']}:")

            centering_col = st.columns([0.62, 0.38])  # st.columns([0.15, 0.70, 0.15])
            with centering_col[0]:
                st.image(f"https://steamcdn-a.akamaihd.net/steam/apps/{quick_stats['app_id']}/header.jpg",
                         use_column_width=True)

            with centering_col[1]:
                # QUERY YOUTUBE FOR A GAMEPLAY VIDEO OF THE GAME
                link = get_gameplay_video(quick_stats['game_name'], GOOGLE_API_TOKEN)

                # DISPLAY THE VIDEO
                st.video(link)
                center_subtitle_col = st.columns([0.25, 0.5, 0.1])
                with center_subtitle_col[1]:
                    st.markdown(f"##### '{quick_stats['game_name']} gameplay' from youtube")
            st.divider()

            pie_chart_cols = st.columns([0.3, 0.35, 0.35])
            with pie_chart_cols[0]:
                st.header(f"Quick stats for {quick_stats['game_name']}")
                st.markdown(f"Overall score: :blue[**{quick_stats['reviewScoreDesc']}**]")
                st.markdown(f"Total reviews: :blue[ **{quick_stats['totalReviews']}**]")
                st.markdown(f"Total Positive Reviews: :blue[**{quick_stats['totalPositive']}**]")
                st.markdown(f"Total Negative Reviews: :blue[**{quick_stats['totalNegative']}**]")
                st.markdown(
                    'Average playtime: :blue[**{:.2f} hours**]'.format(quick_stats['averagePlaytimeFromReviews']))
                st.markdown(
                    'Average Positive Review playtime: :blue[**{:.2f} hours**]'.format(
                        quick_stats['averagePositiveTotalPlaytime']))
                st.markdown(
                    'Average Negative Review playtime: :blue[**{:.2f} hours**]'.format(
                        quick_stats['averageNegativeTotalPlaytime']))
            with pie_chart_cols[1]:
                st.plotly_chart(pie1, use_container_width=True)
            with pie_chart_cols[2]:
                st.plotly_chart(pie2, use_container_width=True)

        # SURVIVAL ANALYSIS SECTION
        with st.container():
            st.header("Survival Analysis")
            col1, col2 = st.columns(2)
            with col1:
                # ON REVIEW
                st.subheader("How Many Hours Did Reviewers Play?")
                st.plotly_chart(average_survival)

                # PROCESS WHETHER OR NOT THE MEDIANS ARE CLOSE
                # cohens_d_value = quick_stats['cohens_d']
                whitney_r_value = quick_stats['r_statistic_at_review']

                median_analysis_text = f"{quick_stats['game_name']}"
                if abs(whitney_r_value) < 0.1:
                    # SMALL DIFFERENCE
                    median_analysis_text += " has a :blue[small] difference between positive and negative playtime"
                elif abs(whitney_r_value) < 0.3:
                    # MEDIUM DIFFERENCE
                    median_analysis_text += " has a :blue[medium] difference between positive and negative playtime"
                elif abs(whitney_r_value) < 0.8:
                    # LARGE DIFFERENCE
                    median_analysis_text += " has a :blue[large] difference between positive and negative playtime"
                else:
                    # SIGNIFICANT DIFFERENCE
                    median_analysis_text += " has a :blue[significant] difference between positive and negative playtime"

                difference_in_medians = quick_stats['medianPositivePlaytime'] - quick_stats['medianNegativePlaytime']

                if difference_in_medians >= 0:
                    median_analysis_text += ", and the playtime at review is skewed :green[positive]."
                else:
                    median_analysis_text += ", and the playtime at review is skewed :red[negative]."

                st.markdown(f"#### {median_analysis_text}")
                st.markdown(f"Mann-Whitney r-value: {whitney_r_value:0.2f}")

                with st.expander("More stats"):
                    st.markdown("###### All Reviews:")
                    mean_col, median_col = st.columns(2)
                    with mean_col:
                        st.markdown(
                            f"Average playtime at review: **{quick_stats['averagePlaytimeOnReview']:0.2f}** hours")
                    with median_col:
                        st.markdown(
                            f"Median playtime at review: **{quick_stats['medianPlaytimeOnReview']:0.2f}** hours")
                    st.divider()
                    pos_review_col, neg_review_col = st.columns(2)
                    with pos_review_col:
                        st.markdown("###### Positive reviews:")
                        st.markdown(
                            f"Average playtime at review: **{quick_stats['averagePositivePlaytime']:0.2f}** hours")
                        st.markdown(
                            f"Median playtime at review: **{quick_stats['medianPositivePlaytime']:0.2f}** hours")
                    with neg_review_col:
                        st.markdown("###### Negative reviews:")
                        st.markdown(
                            f"Average playtime at review: **{quick_stats['averageNegativePlaytime']:0.2f}** hours")
                        st.markdown(
                            f"Median playtime at review: **{quick_stats['medianNegativePlaytime']:0.2f}** hours")

                # EXPLANATIONS WITH MORE INFO
                with st.expander("What does this mean?"):
                    st.markdown(
                        "Mann-Whitney's r-value is a measure of effect size (from -1 to 1) calculated from the U statistic. Simply put, the r-value represents how different the reviewers' playtime is. (Note that it is not related to $ r^{2} $).")
                    st.markdown(
                        "Positively skewed playtime can mean that the game fills a niche, possibly implying that the game is unique, difficult, or quirky in ways that leads critics to abandon it early.")
                    st.markdown(
                        "Similar playtime can depend on how high playtime is and can either mean that the game was boring or mediocre, or that the game is controversial. Critics with high playtime may have been angered or disappointed by updates to the game that affected their enjoyment.")
                    st.markdown(
                        "Negatively skewed playtime means that the experience for players became less enjoyable as time went on, or it can be from changes to the game that the reduced enjoyment of committed players. High negative playtime can also mean that users express disapointment in reviews, but may ultimately continue playing.")
            with col2:
                # POST REVIEW
                st.subheader("How Many Hours Were Played Post-Review?")
                st.plotly_chart(post_completion_survival)

                whitney_r_value_post_review = quick_stats['r_statistic_post_review']

                continued_median_analysis_text = f"{quick_stats['game_name']} has a "
                if abs(whitney_r_value_post_review) < 0.1:
                    # SMALL DIFFERENCE
                    continued_median_analysis_text += ":blue[small]"
                elif abs(whitney_r_value_post_review) < 0.3:
                    # MEDIUM DIFFERENCE
                    continued_median_analysis_text += ":blue[medium]"
                elif abs(whitney_r_value_post_review) < 0.8:
                    # LARGE DIFFERENCE
                    continued_median_analysis_text += ":blue[large]"
                else:
                    # SIGNIFICANT DIFFERENCE
                    continued_median_analysis_text += ":blue[significant]"
                continued_median_analysis_text += " difference between positive and negative playtime"

                average_playtime_post_review = quick_stats['averagePlaytimePostReview']
                median_playtime_post_review = quick_stats['medianPlaytimePostReview']

                # POSSIBLE COMPARE THE DIFFERENCES BETWEEN THIS AND OTHER GAMES?
                continued_median_analysis_text += ". Median playtime post-review: "
                if median_playtime_post_review < 10:
                    continued_median_analysis_text += f":red[{median_playtime_post_review:.0f}] hours."
                elif median_playtime_post_review < 50:
                    continued_median_analysis_text += f":blue[{median_playtime_post_review:.0f}] hours."
                else:
                    continued_median_analysis_text += f":green[{median_playtime_post_review:.0f}] hours."

                st.markdown(f"#### {continued_median_analysis_text}")
                st.markdown(f"Mann-Whitney r-value: {whitney_r_value_post_review:0.2f}")

                with st.expander("More stats"):
                    st.markdown("###### All Reviews:")
                    mean_col, median_col = st.columns(2)
                    with mean_col:
                        st.markdown(
                            f"Average playtime post-review: **{average_playtime_post_review:0.2f}** hours")
                    with median_col:
                        st.markdown(
                            f"Median playtime post-review: **{median_playtime_post_review:0.2f}** hours")
                    st.divider()
                    pos_review_col, neg_review_col = st.columns(2)
                    with pos_review_col:
                        st.markdown("###### Positive reviews:")
                        st.markdown(
                            f"Average post-review playtime: **{quick_stats['averagePositiveContinuedPlaytime']:0.2f}** hours")
                        st.markdown(
                            f"Median post-review playtime: **{quick_stats['medianPositiveContinuedPlaytime']:0.2f}** hours")
                    with neg_review_col:
                        st.markdown("###### Negative reviews:")
                        st.markdown(
                            f"Average post-review playtime: **{quick_stats['averageNegativeContinuedPlaytime']:0.2f}** hours")
                        st.markdown(
                            f"Median post-review playtime: **{quick_stats['medianNegativeContinuedPlaytime']:0.2f}** hours")

                with st.expander("What does this mean?"):
                    st.markdown(
                        "Mann-Whitney's r-value is a measure of effect size (from -1 to 1) calculated from the U statistic. Simply put, the r-value represents how different the reviewers' playtime is. (Note that it is not related to $ r^{2} $).")
                    st.markdown(
                        "Things that may affect playtime post-review include game replayability, number of game updates, and whether or not the game is multiplayer.")
                    st.markdown(
                        "Small/medium differences between positive and negative reviews with high playtime might mean that the game has mixed reception, but has high replayability.")
                    st.markdown(
                        "Large differences between positive and negative reviews with high playtime might mean that the game is highly replayable, but caters to a specific audience. It could also be a sign of a large change in the game that lead reviewers to stop playing")

        # RECOMMENDATION % OVER TIME
        st.divider()
        with st.container():
            # CREATE A FIGURE TO CONTAIN ALL THE OTHER GRAPHS DATA
            combined_trend_graphs = go.Figure()
            st.subheader(
                f"Compare the positive review rates of {quick_stats['game_name']} at different playtimes with other games:")

            compare_columns = st.columns([0.2, 0.6, 0.2])
            with compare_columns[1]:
                trend1, trend2, trend3, trend4, trend5, trend6, trend7, trend8, trend9 = st.tabs(
                    [quick_stats['game_name'], "Elden Ring", "Counter-Strike 2", "Starfield", "No Man's Sky",
                     "Terraria", "NBA 2K24", "Apex Legends", "Combined"])
                color_scheme = px.colors.qualitative.Antique
                with trend1:
                    st.plotly_chart(general_trend, use_container_width=True)
                with trend2:
                    with open('./assets/EldenRingExample.json', 'r') as file:
                        Elden_Ring_chart_data = json.load(file)
                    Elden_Ring_chart_data['data'][0]['line']['color'] = color_scheme[0]
                    ER_fig = go.Figure(data=Elden_Ring_chart_data['data'], layout=Elden_Ring_chart_data['layout'])
                    st.plotly_chart(ER_fig, use_container_width=True)
                with trend3:
                    with open('./assets/CS2Example.json', 'r') as file:
                        CS2_chart_data = json.load(file)
                    CS2_chart_data['data'][0]['line']['color'] = color_scheme[1]
                    CS2_fig = go.Figure(data=CS2_chart_data['data'], layout=CS2_chart_data['layout'])
                    st.plotly_chart(CS2_fig, use_container_width=True)
                with trend4:
                    with open('./assets/StarfieldExample.json', 'r') as file:
                        Starfield_chart_data = json.load(file)
                    Starfield_chart_data['data'][0]['line']['color'] = color_scheme[2]
                    S_fig = go.Figure(data=Starfield_chart_data['data'], layout=Starfield_chart_data['layout'])
                    st.plotly_chart(S_fig, use_container_width=True)
                with trend5:
                    with open('./assets/NoManSkyExample.json', 'r') as file:
                        NoMansSky_chart_data = json.load(file)
                    NoMansSky_chart_data['data'][0]['line']['color'] = color_scheme[3]
                    NMS_fig = go.Figure(data=NoMansSky_chart_data['data'], layout=NoMansSky_chart_data['layout'])
                    st.plotly_chart(NMS_fig, use_container_width=True)
                with trend6:
                    with open('./assets/TerrariaExample.json', 'r') as file:
                        Terraria_chart_data = json.load(file)
                    Terraria_chart_data['data'][0]['line']['color'] = color_scheme[4]
                    Terraria_fig = go.Figure(data=Terraria_chart_data['data'], layout=Terraria_chart_data['layout'])
                    st.plotly_chart(Terraria_fig, use_container_width=True)
                with trend7:
                    with open('./assets/NBA2K24Example.json', 'r') as file:
                        NBA_chart_data = json.load(file)
                    NBA_chart_data['data'][0]['line']['color'] = color_scheme[5]
                    NBA_fig = go.Figure(data=NBA_chart_data['data'], layout=NBA_chart_data['layout'])
                    st.plotly_chart(NBA_fig, use_container_width=True)
                with trend8:
                    with open('./assets/ApexLegendsExample.json', 'r') as file:
                        Apex_chart_data = json.load(file)
                    Apex_chart_data['data'][0]['line']['color'] = color_scheme[6]
                    Apex_fig = go.Figure(data=Apex_chart_data['data'], layout=Apex_chart_data['layout'])
                    st.plotly_chart(Apex_fig, use_container_width=True)
                with trend9:
                    # MODIFY GENERAL TREND GRAPH TO BE IN-LINE WITH THE OTHERS
                    original_trend_data = general_trend.data[0]
                    original_trend_data['name'] = quick_stats['game_name']
                    original_trend_data['x'] = ER_fig.data[0]['x']
                    original_trend_data['line']['color'] = "#ff0000"
                    original_trend_data['showlegend'] = True

                    # COMBINE THE ORIGINAL GRAPH WITH THE NEW ONE
                    combined_trend_graphs.add_trace(original_trend_data)

                    # COMBINE THE REST WITH THE ORIGINAL
                    graph_names = ["Elden Ring", "Counter-Strike 2", "Starfield", "No Man's Sky", "Terraria",
                                   "NBA 2K24", "Apex Legends"]

                    for i, graph_data in enumerate(
                            [ER_fig, CS2_fig, S_fig, NMS_fig, Terraria_fig, NBA_fig, Apex_fig]):
                        line_chart_data_to_add = graph_data.data[0]
                        line_chart_data_to_add['showlegend'] = True
                        line_chart_data_to_add['name'] = graph_names[i]
                        line_chart_data_to_add['line']['color'] = color_scheme[i]
                        combined_trend_graphs.add_trace(line_chart_data_to_add)

                    st.plotly_chart(combined_trend_graphs, use_container_width=True)

        # NETWORK GRAPH SECTION
        st.divider()
        st.header("User similarity analysis")
        st.markdown("**Displays connectivity between the top 7 games in reviewer's libraries**")
        st.markdown(f":red[Excluded {quick_stats['private_profiles']} private profiles from data]")
        tab1, tab2, tab3 = st.tabs(["All", "Positive", "Negative"])
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

        # TOPIC ANALYSIS SECTION
        st.divider()
        with st.container():
            st.header("Topic analysis")
            st.plotly_chart(topics, use_container_width=True)
            st.markdown("#### [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) generated topics:")
            st.markdown(
                "Zephyr is part of the Hugging Face family of models and is a specialized version of the Mystral LLM. The topics generated above are fed to Zephyr to create a unifying category for the words in the topic. Note that accuracy of these topics is not guaranteed, and is highly dependent on the number of input reviews")
            expander_columns = st.columns(5)

            with st.spinner("Using hugging face to generate topics... :hugging_face:"):
                zephyr_generated_topics = []
                for topic in quick_stats['top_words']:
                    zephyr_generated_topics.append(generate_topics(topic, key=HUGGING_API_TOKEN))

            # GET THE COLUMN THAT THE EXPANDER WILL BE PUT IN
            expander_column_index = 0
            for row_index, topics in enumerate(zephyr_generated_topics):

                # IF IT'S AT 5, SET IT BACK TO 0
                if expander_column_index == 5:
                    expander_column_index = 0

                with expander_columns[expander_column_index]:
                    with st.expander(f"###### Topic {row_index + 1}: "):
                        for word in topics:
                            st.markdown(word.lower())

                # INCREMENT THE COLUMN INDEX
                expander_column_index += 1
        st.toast("Topic analysis with Zephyr has finished")
    except Exception as e:
        st.error('Error: Please double check game spelling and try again', icon="🚨️")
        loading_placeholder.empty()
        print(traceback.print_exc())

# # EXAMPLE COMPONENTS
# st.warning('This is a warning', icon="⚠️")

# TODO: MOST PAIRWISE CONNECTIONS AND THEN RECOMMEND GAMES BASED ON THAT
# TODO: COLLABORATIVE FILTERING
# TODO: RAY PYTHON
# https://docs.ray.io/en/latest/ray-core/walkthrough.html
