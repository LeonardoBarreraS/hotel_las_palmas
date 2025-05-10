#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk import download
import nltk
import re
import os

#%%
#importing the dataset
df_sentiment_full=pd.read_csv("../Data/sentiment_dataset_openai_hotel.csv")
df_sen=df_sentiment_full.copy()
df_sen=df_sen.dropna(subset=["suggestions"])


# %%
# --- Cleaning the text data ---
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df_sen["clean_suggestions"]=df_sen["suggestions"].apply(lambda x: [word for word in str(x).split() if word not in stop_words])
df_sen["clean_suggestions"]=df_sen["clean_suggestions"].apply(lambda x: " ".join(x))


# # Filter reviews containing the specific text
# filtered_reviews = df_sen[df_sen["suggestions"].str.contains("Consider adding more color options", case=False, na=False)]

# # Display the filtered reviews
# print(filtered_reviews[["suggestions", "review"]])

#%%

# --- Creating the TF-IDF matrix ---
vectorizer=TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
tfidf=vectorizer.fit_transform(df_sen["clean_suggestions"])

# %%
# ---- Aplicar matriz NMF para encontrar temas (reducción de dimensionalidad)----
n_temas=11
nmf=NMF(n_components=n_temas)

nmf.fit(tfidf)

W=nmf.transform(tfidf) #matriz documento-tema

H=nmf.components_ #matriz tema-palabra

#%%
# --- Mostrar los temas (muestra el vocabulario más relevante de cada tema)---
# vocabulario = vectorizer.get_feature_names_out()

# for idx, topic in enumerate(H):
#     print(f"\n🔹 Tema {idx + 1}:")
#     top_palabras = [vocabulario[i] for i in topic.argsort()[:-6:-1]]
#     print(" - ".join(top_palabras))


# %%
# --- 6. (Opcional) Asignar temas a cada review ---
temas_asignados = W.argmax(axis=1)
valores_asignados = W.max(axis=1)
df_sen["tema"] = temas_asignados
df_sen["valor_tema"] = valores_asignados

#%%
#Themes dictionary
temas_dict ={
    1: "Cuartos con más capacidad de hospedaje",
    2: "Mejorar calidad del aire acondicionado y limpieza de los cuartos",
    3: "Más colores en la decoración de los cuartos",
    4: "Brindar más información sobre los atractivos turísticos",
    5: "Mejorar la calidad de atención del personal",
    6: "Mejorar el menú del desayuno", 
    7: "Mejorar algunos amenities en las áreas comunes"
}

mapping= {0:-1, 1:1, 2:2, 3:-1, 4:3, 5:4, 6:5, 7:6, 8:7, 9:-1, 10:-1}


#Replacing the tema column with the mapping
df_sen["tema"]=df_sen["tema"].replace(mapping)
df_sen["desc_tema"]=df_sen["tema"].replace(temas_dict)
#%%
os.makedirs("../Data", exist_ok=True)
df_sen.to_csv("../Data/sentiment_analysis_suggestions_themes.csv", index=False)

# %%
# Por cada tema, seleccionar los 5 comentarios con mayor "valor_tema"








