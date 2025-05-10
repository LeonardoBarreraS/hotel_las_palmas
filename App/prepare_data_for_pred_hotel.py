#%%
import sys
import os
sys.path.append(os.path.abspath(".."))

from utils_pack import *

#%%


def prepare_data_for_pred():
#%%
    #Load Hotel Data 
    df_hotel=generate_dataframe()


    #Load Sentiment Analysis Data from OpenAI and HuggingFace
    df_sentiment=pd.read_csv("../Data/sentiment_dataset_openai_hotel.csv")

    #Load avg daily sentiment from HuggingFace
    df_dayly_sentiment=pd.read_csv("../Data/sentiment_huggingface_avg_daily_sentiment.csv")
    df_dayly_sentiment["date"]=pd.to_datetime(df_dayly_sentiment["date"])
#%%
    #Merging df_hotel and df_dayly_sentiment
    df_hotel_final=pd.merge(df_hotel, df_dayly_sentiment, how="left", on ="date")
    df_hotel_final["sentiment_numeric"]=df_hotel_final["sentiment_numeric"].fillna(df_hotel_final["sentiment_numeric"].mean())
 
#%%

    #Features List- except date and effective_bookings
    features_list=df_hotel_final.columns.tolist()
    features_list.remove("date")
    features_list.remove("effective_bookings")
#%%  
    return df_hotel_final, features_list

 