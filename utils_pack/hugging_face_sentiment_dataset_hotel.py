#%%
from datasets import load_dataset
from transformers import pipeline
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#%%
ds=load_dataset("ashraq/hotel-reviews")
print(ds)
# %%
df_reviews=ds['train']["review"]
sample_reviews = random.sample(df_reviews, 5000)
# %%
#Building review  dataset
import pandas as pd

df_reviews=pd.DataFrame(sample_reviews, columns=["review"])
df_reviews["review"]=df_reviews["review"].astype(str)

# %%
#Importing sentiment analysis pipeline

#pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")


#%%

df_reviews["sentiment"]=df_reviews["review"].apply(lambda x: pipe(x)[0]["label"])
# %%
os.makedirs("../Data", exist_ok=True)
df_reviews.to_csv("../Data/sentiment_dataset_huggingface_hotel.csv", index=False)

print("✅ Datos generados correctamente.")
# %%
#Loading the reviews dataset
reviews_df=pd.read_csv("../Data/sentiment_dataset_huggingface_hotel.csv")
print(reviews_df.shape)
#%%
#Creating random dates for the reviews
end_date = (datetime.today() - timedelta(days=3)).date()
start_date = (datetime.today() - timedelta(days=733)).date()

date_range = (end_date - start_date).days

reviews_df["date"] = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(len(reviews_df))]


# %%
#Cuantifying the sentiment of the reviews
sentiment_mapping={
    "POSITIVE": 1,
    "NEGATIVE": -1,
    "NEUTRAL": 0
}

reviews_df["sentiment_numeric"]=reviews_df["sentiment"].apply(lambda x: sentiment_mapping[x])

#Grouping by date and calculating the average sentiment
daily_sentiment=reviews_df.groupby("date")["sentiment_numeric"].mean().reset_index()

print(daily_sentiment.head())
#saving the daily sentiment to a csv file
os.makedirs("../Data", exist_ok=True)
daily_sentiment.to_csv("../Data/sentiment_huggingface_avg_daily_sentiment.csv", index=False)

print("✅ Datos generados correctamente.")


# %%
