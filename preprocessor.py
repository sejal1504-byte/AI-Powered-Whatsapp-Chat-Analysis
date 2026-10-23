import re
import pandas as pd


def preprocess(data):

    # WhatsApp date pattern
    pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s-\s'


    messages = re.split(pattern, data)


    dates = []
    msg = []


    for i in range(1, len(messages)-1, 3):

        date = messages[i]
        time = messages[i+1]
        message = messages[i+2]

        dates.append(date + " " + time)
        msg.append(message)



    df = pd.DataFrame(
        {
            "user_message": msg,
            "date": dates
        }
    )


    if df.empty:
        return df



    # Date conversion

    df['date'] = pd.to_datetime(
        df['date'],
        dayfirst=True,
        errors='coerce'
    )


    df.dropna(
        subset=['date'],
        inplace=True
    )



    users=[]
    messages=[]


    for message in df['user_message']:


        user = re.split(
            r'([\w\W]+?):\s',
            message,
            maxsplit=1
        )


        if len(user)>2:

            users.append(
                user[1]
            )

            messages.append(
                user[2]
            )


        else:

            users.append(
                "group_notification"
            )

            messages.append(
                user[0]
            )



    df['user']=users

    df['message']=messages



    df.drop(
        columns=['user_message'],
        inplace=True
    )



    # =============================
    # Extra Date Features
    # =============================


    df['only_date'] = df['date'].dt.date

    df['year'] = df['date'].dt.year

    df['month_num'] = df['date'].dt.month

    df['month'] = df['date'].dt.month_name()

    df['day'] = df['date'].dt.day

    df['day_name'] = df['date'].dt.day_name()

    df['hour'] = df['date'].dt.hour

    df['minute'] = df['date'].dt.minute



    # =============================
    # REQUIRED FOR HEATMAP
    # =============================


    period=[]


    for hour in df['hour']:

        if hour == 23:

            period.append(
                str(hour)+"-00"
            )

        elif hour == 0:

            period.append(
                "00-1"
            )

        else:

            period.append(
                str(hour)+"-"+str(hour+1)
            )



    df['period']=period



    return df