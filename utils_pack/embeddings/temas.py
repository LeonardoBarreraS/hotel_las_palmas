#%%
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import pandas as pd
import csv
from collections import Counter

#%%
#Creating the a Client for the ChromaDB
current_dir = os.getcwd()
client = chromadb.PersistentClient(path=current_dir)
#%%
#Creating a collection
# collection= client.create_collection(
#     name="temas", 
#     embedding_function=OpenAIEmbeddingFunction(
#         model_name="text-embedding-ada-002",
#     )
# )

client.list_collections()
#%%
#Dict of strategic variables in a Hotel
variables = {
    "room_cleanliness": "The level of cleanliness and hygiene maintained in the hotel's rooms, ensuring a comfortable stay for guests.",
    "staff_friendliness": "The courtesy and willingness of hotel staff to address guests' needs and concerns.",
    "location_accessibility": "The convenience of the hotel's location relative to popular attractions, transportation, and services.",
    "price_quality_ratio": "The perceived value of the hotel's services and facilities in relation to the price paid.",
    "amenity_quality": "The quality and availability of facilities such as Wi-Fi, gym, pool, and dining options that enhance the guest experience.",
    "service_personalization": "The hotel's ability to tailor its services and offerings to individual customer preferences.",
    "technological_facilities": "The availability and efficiency of technologies such as mobile apps, automatic check-in, and digital assistance.",
    "room_size": "The spaciousness and overall size of the hotel's rooms, contributing to guest comfort and satisfaction."
}

# Traducción de las claves al español
translations = {
    "room_cleanliness": "limpieza_de_habitaciones",
    "staff_friendliness": "amabilidad_del_personal",
    "location_accessibility": "accesibilidad_de_ubicación",
    "price_quality_ratio": "relación_precio_calidad",
    "amenity_quality": "calidad_de_servicios",
    "service_personalization": "personalización_del_servicio",
    "technological_facilities": "instalaciones_tecnológicas",
    "room_size": "tamaño_de_habitación"
}

english_keys = list(variables.keys())

#------------------------------------>>>>>>>>>>>>>>>>>>>>>>
# %%
#Adding the data to the collection
# collection=client.get_collection("temas", embedding_function=OpenAIEmbeddingFunction(
#         model_name="text-embedding-ada-002",
#         api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#     ))

#%%
#contar documentos en la colección
print(f"Documentos en la colección: {collection.count()}")

#------------------------------------>>>>>>>>>>>>>>>>>>>>>>
#%%
#Creating the documents to be added to the collection. Key + Value from the variables dict. 
docs=[key+ ": " + value for key, value in variables.items()]
print(docs)

#%%
collection.upsert(
    ids=list(variables.keys()),
    documents=docs
)
#%%
# Retrieve the document and its embedding for "room_size"
result = collection.get("room_size", include=["embeddings", "documents"])
print(result)

#%%
#Extraer los reviews negativos de la base de datos
id=[]
texts=[]

file_path = "I:\Mi unidad\Data Science\Projects\Hotel Booking Forcast & Sentiment Analysis\Data\sentiment_dataset_huggingface_hotel.csv"

with open(file_path) as csvfile:
    reader=csv.DictReader(csvfile)
    print(type(reader))
    for i, row in enumerate(reader):
        if row["sentiment"] == "NEGATIVE":
            id.append(str(i))
            texts.append(row["review"])
            

texts = [str(text) for text in texts]

  # Asegúrate de que todos los textos sean cadenas
invalid_texts= [t for t in texts if not isinstance(t, str) or t.strip()==""]
print(f"# Invalid texts: {len(invalid_texts)}")


#%%
# Consultar la colección y obtener los 2 vecinos más cercanos para cada texto
batch_size = 100
results = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    response = collection.query(
        query_texts=batch,
        n_results=2  # Por ejemplo, quieres los 5 vecinos más cercanos
    )
    results.append(response)

#%%
#Unificar los resultados en un solo diccionario
dict={}
dict["id"]=[]
dict["ids"]=[]
dict["distances"]=[]
dict["documents"]=[]
for i in range(len(results)):
    for j in range(len(results[i]["ids"])):
        dict["id"].append(id[i])
        dict["ids"].append(results[i]["ids"][j])
        dict["distances"].append(results[i]["distances"][j])
        dict["documents"].append(results[i]["documents"][j])

#%%
# Crear una lista de tuplas (query_idx, match_idx, distance) para los matches válidos

MAX_DISTANCE_THRESHOLD = 0.45  # máximo permitido para considerar un match válido
# Lista para acumular las parejas (query_idx, match_idx, distance)
valid_matches = []

for query_idx, distance in enumerate(dict["distances"]):
    lower_distance = distance[0]  # Distancia del primer match
    if lower_distance <= MAX_DISTANCE_THRESHOLD:
        # Si la distancia es válida, agregar la pareja (query_idx, match_idx, distance)
        valid_matches.append((id[query_idx], texts[query_idx], dict["ids"][query_idx][0], lower_distance))
#%%
print(f"Total valid matches: {len(valid_matches)}")     

#%%
# Crear el DataFrame con las columnas solicitadas
df_matches = pd.DataFrame(valid_matches, columns=["query_id", "review_text", "matched_variable_id", "distance"])

# Agregar la descripción de la variable usando el diccionario 'variables'
df_matches["matched_variable_description"] = df_matches["matched_variable_id"].map(variables)

# Mostrar las primeras filas del DataFrame
print(df_matches.head())

#%%
data_sent=pd.read_csv(file_path)
print(data_sent.head())
#%%
#Merge the data_sent dataframe with df_matches on the index and the query_id column
data_sent["query_id"] = data_sent.index.astype(str)
data_sent = data_sent.merge(df_matches, on="query_id", how="left")
#print(data_sent.head())

data_sent[data_sent["sentiment"] == "NEGATIVE"].head(50)


#%%
os.makedirs("I:\Mi unidad\Data Science\Projects\Hotel Booking Forcast & Sentiment Analysis\Data", exist_ok=True)
data_sent.to_csv("I:\Mi unidad\Data Science\Projects\Hotel Booking Forcast & Sentiment Analysis\Data\sentiment_dataset_huggingface_hotel2.csv", index=False)

print("✅ Datos generados correctamente.")


#%%