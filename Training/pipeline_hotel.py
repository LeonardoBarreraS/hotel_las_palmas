#%%
import sys
import os
sys.path.append(os.path.abspath(".."))

import joblib
from sklearn.preprocessing import MinMaxScaler

from utils_pack import *
from Training import *
#from utils_pack.dataset_hotel import generate_dataframe

from sklearn.model_selection import train_test_split
from scipy.special import expit

import mlflow
import mlflow.pytorch

#%%
#Load Hotel Data 
df_hotel=generate_dataframe()
print(df_hotel.shape)
print(df_hotel.info())

#%%
#Load avg daily sentiment from HuggingFace
df_dayly_sentiment=pd.read_csv("../Data/sentiment_huggingface_avg_daily_sentiment.csv")
df_dayly_sentiment["date"]=pd.to_datetime(df_dayly_sentiment["date"])
print(df_dayly_sentiment.head())
print(df_dayly_sentiment.info())

# %%
#Merging df_hotel and df_dayly_sentiment
df_hotel_final=pd.merge(df_hotel, df_dayly_sentiment, how="left", on ="date")
df_hotel_final["sentiment_numeric"]=df_hotel_final["sentiment_numeric"].fillna(df_hotel_final["sentiment_numeric"].mean())
print(df_hotel_final.head())

#%%
os.makedirs("../Data", exist_ok=True)
df_hotel_final.to_csv("../Data/dataset_hotel_final.csv", index=False)

# %%
#Features List- except date and effective_bookings
features_list=df_hotel_final.columns.tolist()
features_list.remove("date")
features_list.remove("effective_bookings")
print(features_list)

#%%
#Scale with MinMaxScaler the features
os.makedirs("../utils_pack/scaler", exist_ok=True)
scaler=MinMaxScaler()
df_hotel_final[features_list]=scaler.fit_transform(df_hotel_final[features_list])
joblib.dump(scaler, "../utils_pack/scaler/scaler_hotel.pkl")

#%%
#Extracting X and y dataframes and splitting into train and test
y = df_hotel_final['effective_bookings']
X = df_hotel_final[features_list]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
input_dim=X_train.shape[1]
print(input_dim)
print(X_train.shape)
print(X_test.shape)


# %%
#Creating the Dataloaders
seq_len=90
n_predictions=30

train_data_loader=generate_dataloader(X_train, y_train, features_list, seq_len, n_predictions)
test_data_loader=generate_dataloader(X_test, y_train, features_list, seq_len, n_predictions)

print(len(train_data_loader))
print(len(test_data_loader))
# %%
#Instantiating the model
model=HotelPredModel(input_dim, n_predictions)

# %%
#MlFlow setup
mlflow.set_tracking_uri("../Model/model/mlruns")
experiment_name= "hotel_booking_pred"

try:
    mlflow.create_experiment(experiment_name)
except Exception as e:
    print(f"Experiment {experiment_name} already exists. Using the existing experiment.")

#%%
mlflow.set_experiment(experiment_name)
with mlflow.start_run() as run:
    pred=train_model_hotelpred(model, train_data_loader)
    mlflow.pytorch.log_model(model, "model")
    pred_mse=eval_hotel_model(pred, test_data_loader)
    pred_mse_perday=eval_hotel_model_perday(pred, test_data_loader)
    mlflow.log_metric("mse", pred_mse)
    mlflow.log_metric("mse_per_day", pred_mse_perday.mean().item())
    print("Training and evaluation completed")


# %%
