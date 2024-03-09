# THE FOLLOWING WAS WRITTEN IN COLLABORATION WITH SAMSAQ FOR A PROJECT IN ASU CSE 445
# https://github.com/samsaq
# https://github.com/samsaq/CSE445-Term-Project/blob/main/services/steamWrapper.py
import requests
from urllib.parse import quote

debug = True


def get_Reviews(appid=1091500, reviewType="all", reviewNum=300):
    # setup the query parameters
    filter_reviews = "recent"
    language = "english"
    day_range = 365
    cursor = "*"
    purchase_type = "steam"
    reviewJSONs = []

    # if the reviewNum is greater than 100, we'll need to make multiple requests
    # if it is greater, we'll do requests of 100 until we get to the last request, which will be the remainder
    # we'll use reviewsRemaining to keep track of how many reviews we have left to get
    # if it is less, we'll just do one request

    # get the initial number of reviews per request
    num_per_page = 1
    reviewsRemaining = int(reviewNum)
    if (int(reviewNum) > 100):
        num_per_page = 100
    # make the requests, we'll need to use the cursor from this request to get the next page, and will get 3 pages for 300 reviews for now
    # review request loop
    while reviewsRemaining > 0:
        # modify the num_per_page if we have less than 100 reviews remaining
        if (reviewsRemaining < 100):
            num_per_page = reviewsRemaining

        requestUrl = "https://store.steampowered.com/appreviews/" + str(appid) + "?json=1" + "&filter=" + str(
            filter_reviews) + "&language=" + str(language) + "&day_range=" + str(day_range) + "&cursor=" + str(
            cursor) + "&review_type=" + str(reviewType) + "&purchase_type=" + str(
            purchase_type) + "&num_per_page=" + str(num_per_page)
        # make the request
        response = requests.get(requestUrl)
        # check for errors, if the status code is not 200, we have an error
        if response.status_code != 200:
            # return the error code
            return response.status_code
        # get the data from the response & update variables as needed
        jsonData = response.json()
        reviewJSONs.append(jsonData)
        cursor = quote(jsonData["cursor"])
        reviewsRemaining -= num_per_page

    # now to parse the json data and reformat it for our needs

    # first get overall details, the "query_summary" key has the overall details, we need the "review_score_desc", "total_positive", "total_negative", and "total_reviews"
    # just get this from the first json, since it will be the same for all of them
    overallDetails = reviewJSONs[0]["query_summary"]
    reviewScoreDesc = overallDetails["review_score_desc"]
    totalPositive = overallDetails["total_positive"]
    totalNegative = overallDetails["total_negative"]
    totalReviews = overallDetails["total_reviews"]
    overallDetailsDict = {"reviewScoreDesc": reviewScoreDesc, "totalPositive": totalPositive,
                          "totalNegative": totalNegative, "totalReviews": totalReviews}

    # now to get the review data, the review data is in the "reviews" key, which is a list of dictionaries
    reviews = []
    reviews.append(overallDetailsDict)
    # for each review, we'll need to get the "review" which is the review text, the "voted_up" which is true for positive, and false for negative, and the "recommendationid"
    # we'll need to get the "author" key which has a steamid, and a playtime_forever, playtime_at_review, and playtime_last_two_weeks, and last_played we'll want
    # also want to compare the timestamp_created and timestamp_updated to see if the review was updated
    # the rest we don't need
    # get all of that together and put it in a dictionary, then append it to the reviews list

    for reviewJSON in reviewJSONs:
        for review in reviewJSON["reviews"]:
            reviewText = review["review"]
            reviewPositive = review["voted_up"]
            reviewRecommendationID = review["recommendationid"]
            reviewTimestampCreated = review["timestamp_created"]
            reviewTimestampUpdated = review["timestamp_updated"]
            if (reviewTimestampCreated != reviewTimestampUpdated):
                reviewEdited = True
            else:
                reviewEdited = False
            reviewAuthor = review["author"]
            reviewAuthorSteamID = reviewAuthor["steamid"]
            reviewPlaytimeForever = reviewAuthor["playtime_forever"]
            reviewPlaytimeAtReview = 0
            if 'playtime_at_review' in reviewAuthor.keys():
                reviewPlaytimeAtReview = reviewAuthor["playtime_at_review"]
            reviewPlaytimeLastTwoWeeks = reviewAuthor["playtime_last_two_weeks"]
            reviewLastPlayed = reviewAuthor["last_played"]
            numberOfReviews = reviewAuthor["num_reviews"]
            reviewDict = {"reviewText": reviewText, "reviewPositive": reviewPositive, "reviewEdited": reviewEdited,
                          "reviewRecommendationID": reviewRecommendationID, "reviewAuthorSteamID": reviewAuthorSteamID,
                          "reviewPlaytimeForever": reviewPlaytimeForever,
                          "reviewPlaytimeAtReview": reviewPlaytimeAtReview,
                          "reviewPlaytimeLastTwoWeeks": reviewPlaytimeLastTwoWeeks,
                          "reviewLastPlayed": reviewLastPlayed,
                          "numberOfReviews": numberOfReviews}
            reviews.append(reviewDict)

    return reviews


# QUERY YOUTUBE FOR VIDEOS:
def get_gameplay_video(game_name, key):
    endpoint = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=20&q={game_name} gameplay&type=video&key={key}&safeSearch=strict"
    response = requests.get(endpoint)

    video_results = response.json()
    print(video_results)
    try:
        video_id = video_results['items'][0]['id']['videoId']
    except Exception as e:
        raise "Error with the youtube video"
    return f"https://www.youtube.com/watch?v={video_id}"
