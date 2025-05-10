#%%
from bertopic import BERTopic
import pandas as pd


#%%
#Extracting suggestions from the reviews
df_sentiment_full=pd.read_csv("../Data/sentiment_dataset_openai_hotel.csv")
df_sen=df_sentiment_full.copy()
df_sen=df_sen.dropna(subset=["suggestions"])
textos=df_sen["suggestions"].tolist()

# %%

#Instantiating the BERTopic model and fitting-transforming the data
topic_model =BERTopic(language="english")
temas, probabilidades = topic_model.fit_transform(textos)
# %%
#Assigning topics and probabilities of the topics to the dataframe
print(f"Unique values in 'temas': {len(set(temas))}")

df_sen["temas"]=temas

df_sen["probabilidades"]=probabilidades.tolist()
print(df_sen.head())

#%%
topic_model.reduce_topics(textos, nr_topics=10)

# %%
#Reducing the number of topics to 10
topic_info = topic_model.get_topic_info()
temas, probabilidades = topic_model.transform(textos)
#%%
# Reassigning temas y probabilidades to the dataframe
print(type(topic_info))
print(topic_info[["Name", "Representation", "Representative_Docs"]])

list_topics = topic_info["Name"].tolist()
print(list_topics)

print(f"Unique values in 'temas': {len(set(temas))}")

df_sen["temas"]=temas

df_sen["probabilidades"]=probabilidades.tolist()
print(df_sen.head())
# %%
topic_model.visualize_topics()
# %%
