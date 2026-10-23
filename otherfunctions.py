from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


extract = URLExtract()



# ==================================================
# STATS
# ==================================================

def fetch_stats(selected_user, df):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]


    num_messages = df.shape[0]


    words = []

    for message in df['message']:
        words.extend(
            str(message).split()
        )


    media = df[
        df['message'].str.contains(
            "Media omitted",
            na=False
        )
    ].shape[0]


    links = []

    for message in df['message']:
        links.extend(
            extract.find_urls(
                str(message)
            )
        )


    return (
        num_messages,
        len(words),
        media,
        len(links)
    )




# ==================================================
# MOST BUSY USERS
# ==================================================

def most_busy_users(df):

    x = df['user'].value_counts().head()


    percent = round(
        (df['user'].value_counts()/df.shape[0])*100,
        2
    )


    new_df = percent.reset_index()


    new_df.columns = [
        "name",
        "percent"
    ]


    return x,new_df




# ==================================================
# WORD CLOUD
# ==================================================

def create_wordcloud(selected_user, df):


    if selected_user != "Overall":

        df = df[
            df['user']==selected_user
        ]



    temp = df[
        df['user']!="group_notification"
    ]



    temp = temp[
        ~temp['message'].str.contains(
            "Media omitted",
            na=False
        )
    ]



    if temp.empty:

        return WordCloud().generate(
            "No Data"
        )



    try:

        with open(
            "stop_hinglish.txt",
            "r",
            encoding="utf-8"
        ) as f:

            stop_words=f.read()


    except:

        stop_words=""



    def remove_stop_words(message):

        result=[]


        for word in str(message).lower().split():

            if word not in stop_words:

                result.append(word)


        return " ".join(result)



    temp=temp.copy()


    temp['message']=temp['message'].apply(
        remove_stop_words
    )



    wc=WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color="white"
    )



    return wc.generate(
        temp['message'].str.cat(
            sep=" "
        )
    )

# ==================================================
# COMMON WORDS
# ==================================================

def most_common_words(selected_user, df):


    if selected_user != "Overall":

        df = df[
            df['user']==selected_user
        ]



    temp = df[
        df['user']!="group_notification"
    ]



    words=[]


    for message in temp['message']:

        for word in str(message).lower().split():

            words.append(word)



    if len(words)==0:

        return pd.DataFrame()



    return pd.DataFrame(
        Counter(words).most_common(20)
    )





# ==================================================
# EMOJI ANALYSIS
# ==================================================

def emoji_helper(selected_user, df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    emojis=[]


    for message in df['message']:

        emojis.extend(
            [
                c for c in str(message)
                if c in emoji.EMOJI_DATA
            ]
        )



    if len(emojis)==0:

        return pd.DataFrame()



    return pd.DataFrame(
        Counter(emojis).most_common()
    )





# ==================================================
# MONTHLY TIMELINE
# ==================================================

def monthly_timeline(selected_user, df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    timeline = (
        df.groupby(
            [
                'year',
                'month_num',
                'month'
            ]
        )
        .count()['message']
        .reset_index()
    )



    timeline['time'] = (
        timeline['month']
        +
        "-"
        +
        timeline['year'].astype(str)
    )


    return timeline





# ==================================================
# DAILY TIMELINE
# ==================================================

def daily_timeline(selected_user, df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    daily = (
        df.groupby(
            'only_date'
        )
        .count()['message']
        .reset_index()
    )


    return daily





# ==================================================
# ACTIVITY MAP
# ==================================================

def week_activity_map(selected_user,df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    return df['day_name'].value_counts()





def month_activity_map(selected_user,df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    return df['month'].value_counts()





# ==================================================
# WEEKLY HEATMAP (FIXED)
# ==================================================

def activity_heatmap(selected_user,df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    required_columns=[
        'day_name',
        'period',
        'message'
    ]



    for col in required_columns:

        if col not in df.columns:

            return pd.DataFrame()



    heatmap = df.pivot_table(

        index='day_name',

        columns='period',

        values='message',

        aggfunc='count'

    ).fillna(0)



    return heatmap

# ==================================================
# AI SENTIMENT ANALYSIS
# TextBlob
# ==================================================

def sentiment_analysis(selected_user, df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    sentiments=[]



    for message in df['message']:


        polarity = TextBlob(
            str(message)
        ).sentiment.polarity



        if polarity > 0:

            sentiments.append(
                "Positive"
            )


        elif polarity < 0:

            sentiments.append(
                "Negative"
            )


        else:

            sentiments.append(
                "Neutral"
            )



    return pd.Series(
        sentiments
    ).value_counts()





# ==================================================
# AI TOPIC DETECTION
# TF-IDF + LDA
# ==================================================

def topic_detection(selected_user, df):


    if selected_user!="Overall":

        df=df[
            df['user']==selected_user
        ]



    temp=df[
        df['user']!="group_notification"
    ]



    temp=temp[
        ~temp['message'].str.contains(
            "Media omitted",
            na=False
        )
    ]



    # minimum messages required

    if len(temp)<5:

        return pd.DataFrame()



    messages = (
        temp['message']
        .astype(str)
        .tolist()
    )



    try:


        # TF-IDF

        vectorizer = TfidfVectorizer(

            max_features=500,

            stop_words='english'

        )



        tfidf_matrix = vectorizer.fit_transform(
            messages
        )



        # LDA Model

        lda = LatentDirichletAllocation(

            n_components=3,

            random_state=42

        )



        lda.fit(
            tfidf_matrix
        )



        words = (
            vectorizer
            .get_feature_names_out()
        )



        topics=[]



        for index,topic in enumerate(
            lda.components_
        ):



            top_words=[

                words[i]

                for i in topic.argsort()[-10:]

            ]



            topics.append(

                {

                "Topic":
                f"Topic {index+1}",


                "Keywords":
                ", ".join(top_words)

                }

            )



        return pd.DataFrame(
            topics
        )



    except Exception:

        return pd.DataFrame()