import streamlit as st
import preprocessor
import otherfunctions
import matplotlib.pyplot as plt
import seaborn as sns


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI WhatsApp Chat Analyzer",
    layout="wide"
)


st.sidebar.title("🤖 AI WhatsApp Chat Analyzer")


uploaded_file = st.sidebar.file_uploader(
    "Upload WhatsApp Chat (.txt)"
)



if uploaded_file is not None:


    # ================= READ FILE =================

    bytes_data = uploaded_file.getvalue()

    data = bytes_data.decode(
        "utf-8",
        errors="ignore"
    )


    # ================= PREPROCESS =================

    try:

        df = preprocessor.preprocess(data)


    except Exception as e:

        st.error(
            f"Preprocessing Error: {e}"
        )

        st.stop()



    st.subheader("Dataset Preview")


    st.write(
        "Rows & Columns:",
        df.shape
    )


    st.write(
        "Columns:",
        df.columns.tolist()
    )


    st.dataframe(
        df.head()
    )



    if df.empty:

        st.error(
            "No messages found. Check WhatsApp export format."
        )

        st.stop()



    # ================= USER LIST =================


    user_list = df['user'].unique().tolist()


    if "group_notification" in user_list:

        user_list.remove(
            "group_notification"
        )


    user_list.sort()

    user_list.insert(
        0,
        "Overall"
    )



    selected_user = st.sidebar.selectbox(
        "Analyze User",
        user_list
    )



    if st.sidebar.button(
        "Show Analysis"
    ):



        # ================= STATS =================


        num_messages, words, media, links = (
            otherfunctions.fetch_stats(
                selected_user,
                df
            )
        )


        st.title(
            "📊 Top Statistics"
        )


        c1,c2,c3,c4 = st.columns(4)


        c1.metric(
            "Messages",
            num_messages
        )

        c2.metric(
            "Words",
            words
        )

        c3.metric(
            "Media",
            media
        )

        c4.metric(
            "Links",
            links
        )



        # ================= MONTHLY =================


        st.title(
            "📅 Monthly Timeline"
        )


        timeline = otherfunctions.monthly_timeline(
            selected_user,
            df
        )


        if not timeline.empty:


            fig,ax = plt.subplots()

            ax.plot(
                timeline['time'],
                timeline['message']
            )


            plt.xticks(
                rotation=45
            )


            st.pyplot(fig)



        # ================= DAILY =================


        st.title(
            "Daily Timeline"
        )


        daily = otherfunctions.daily_timeline(
            selected_user,
            df
        )


        if not daily.empty:


            fig,ax = plt.subplots()


            ax.plot(
                daily['only_date'],
                daily['message']
            )


            plt.xticks(
                rotation=45
            )


            st.pyplot(fig)



        # ================= ACTIVITY =================


        st.title(
            "🔥 Activity Map"
        )


        busy_day = otherfunctions.week_activity_map(
            selected_user,
            df
        )


        if len(busy_day)>0:


            fig,ax = plt.subplots()


            ax.bar(
                busy_day.index,
                busy_day.values
            )


            plt.xticks(
                rotation=45
            )


            st.pyplot(fig)



        # ================= HEATMAP =================


        st.title(
            "📌 Weekly Activity Heatmap"
        )


        heatmap = otherfunctions.activity_heatmap(
            selected_user,
            df
        )


        if not heatmap.empty:


            fig,ax = plt.subplots()


            sns.heatmap(
                heatmap,
                ax=ax
            )


            st.pyplot(fig)


        else:

            st.info(
                "Heatmap not available"
            )



        # ================= BUSY USERS =================


        if selected_user == "Overall":


            st.title(
                "Most Busy Users"
            )


            x,new_df = otherfunctions.most_busy_users(
                df
            )


            fig,ax = plt.subplots()


            ax.bar(
                x.index,
                x.values
            )


            plt.xticks(
                rotation=45
            )


            st.pyplot(fig)


            st.dataframe(
                new_df
            )



        # ================= WORD CLOUD =================


        st.title(
            "☁ Word Cloud"
        )


        try:

            wc = otherfunctions.create_wordcloud(
                selected_user,
                df
            )


            fig,ax = plt.subplots()


            ax.imshow(
                wc
            )


            ax.axis(
                "off"
            )


            st.pyplot(fig)



        except Exception as e:

            st.warning(
                f"WordCloud Error: {e}"
            )



        # ================= COMMON WORDS =================


        st.title(
            "Most Common Words"
        )


        common_words = otherfunctions.most_common_words(
            selected_user,
            df
        )


        if not common_words.empty:


            fig,ax = plt.subplots()


            ax.barh(
                common_words[0],
                common_words[1]
            )


            st.pyplot(fig)



        # ================= EMOJI =================


        st.title(
            "😀 Emoji Analysis"
        )


        emoji_df = otherfunctions.emoji_helper(
            selected_user,
            df
        )


        if not emoji_df.empty:


            st.dataframe(
                emoji_df
            )



        # ================= SENTIMENT AI =================


        st.title(
            "🤖 AI Sentiment Analysis"
        )


        sentiment = otherfunctions.sentiment_analysis(
            selected_user,
            df
        )


        if not sentiment.empty:


            col1,col2 = st.columns(2)


            with col1:

                st.dataframe(
                    sentiment
                )


            with col2:


                fig,ax = plt.subplots()


                ax.pie(
                    sentiment.values,
                    labels=sentiment.index,
                    autopct="%0.2f%%"
                )


                st.pyplot(fig)



        # ================= TOPIC AI =================


        st.title(
            "🧠 AI Topic Detection (TF-IDF + LDA)"
        )


        try:


            topics = otherfunctions.topic_detection(
                selected_user,
                df
            )


            if not topics.empty:


                st.success(
                    "AI detected conversation topics"
                )


                st.dataframe(
                    topics,
                    use_container_width=True
                )


            else:


                st.info(
                    "Not enough data for topic detection"
                )


        except Exception as e:


            st.error(
                f"Topic Detection Error: {e}"
            )



else:


    st.info(
        "Upload WhatsApp exported txt file to start AI analysis"
    )