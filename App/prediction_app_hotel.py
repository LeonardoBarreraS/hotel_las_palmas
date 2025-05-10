#%%
import streamlit as st
from prepare_data_for_pred_hotel import prepare_data_for_pred
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import torch
import mlflow
import mlflow.pytorch
import sys
import os

# Agrega la carpeta donde está model_hotel.py ANTES de importarlo
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Training")))

from utils_pack import *
import numpy as np

# Importa explícitamente la clase usada para el modelo
from model_hotel import HotelPredModel  # Asegúrate que model_hotel.py está en Training y la clase se llama así

#%%

def prediction():
    #%%
    # Load the data
    df_hotel_final, features_list = prepare_data_for_pred()

    df_hotel_final["date"] = pd.to_datetime(df_hotel_final["date"])
    #ordering by date from the oldest to the most recent
    df_hotel_final=df_hotel_final.sort_values(by="date", ascending=True)

   #%%
    import plotly.express as px

    # Filtrar para mostrar solo los últimos 3 meses
    last_date = df_hotel_final["date"].max()
    three_months_ago = last_date - pd.DateOffset(months=3)
    df_last_3_months = df_hotel_final[df_hotel_final["date"] >= three_months_ago]
    st.write("")
    st.write("")
    # Centrar el gráfico usando columnas vacías a los lados, pero dando más ancho al centro
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        fig = px.line(
            df_last_3_months,
            x="date",
            y="effective_bookings",
            title="<b>Reservas efectivas diarias de los últimos 3 meses</b>",
            labels={"date": "Date", "effective_bookings": "Effective Bookings"},
            template="plotly_white"
        )
        fig.update_layout(
            title={
                'text': "<b>Reservas efectivas diarias de los últimos 3 meses</b>",
                'x': 0.5,
                'xanchor': 'center'
            },
            width=900,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.write("")

    st.markdown(
    """
    <div style='background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;'>
        <p style='margin: 0; color: #2C3E50; font-size: 1em;'>
        Como puedes ver en el gráfico, las reservas efectivas fluctuan a lo largo del tiempo, lo que puede ser influenciado por diversos factores como eventos locales, clima, promociones, etc. 
        <br>
        <br>Sin embargo, con la ayuda de un modelo de predicción, entrando con datos de más de 2 años de operación, podemos anticipar el número de reservas efectivas para los próximos 30 días.
        <br>
        <br>Esto lo hacemos utilizando un modelo de <b>redes neuronales LSTM</b>, que es especialmente eficaz para datos de series temporales.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
        )
#%%

    #creating the features list in spanish
    features_list_spanish = {
        "temperature": "Temperatura",
        "is_holiday": "Es festivo?",
        "is_weekend": "Es fin de semana?",
        "day_of_week": "Día de la semana",
        "month": "Mes del año",
        "week_of_year": "Semana del año",
        "promotion": "Promoción?",
        "sentiment_numeric": "Promedio de sentimiento",
        "avg_room_price": "Precio promedio de habitación",
        "local_event": "Evento local?",
        "occupancy_rate_prev_day": "Ocupación día previo",
        "cancellation_rate": "Tasa de cancelación",
        "bookings": "Reservas",
        "effective_bookings": "Reservas efectivas"
    }

    features_spanish=list(features_list_spanish.values())

    #%%
    st.write("")
    st.write("")
    with st.expander("12 Variables medidas en los datos de entrenamiento del modelo durante los últimos 2 años"):
        st.markdown(
            "\n".join([f"- {feature}" for feature in features_spanish])
        )

    with st.expander("Muestra de 10 días del set de entrenamiento, con sus respectivas variables y la cantidad efectiva de reservas"):
        st.dataframe(df_hotel_final.sample(10, random_state=42))
    
    #%%
    scaler = joblib.load("../utils_pack/scaler/scaler_hotel.pkl")
    #df_hotel_final = df_hotel_final.iloc[-180:, ]
    print(df_hotel_final.shape)
    print(df_hotel_final.head(20))
    #%%
    df_hotel_final[features_list] = scaler.transform(df_hotel_final[features_list])
    X = df_hotel_final[features_list]
    print(X.shape)
    #%%
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    print(X_tensor.shape)
    #%%
    # Suponiendo que tu modelo fue entrenado con secuencias de longitud seq_len (por ejemplo, 30)
    seq_len = 90  # Ajusta esto según tu entrenamiento

    # Tomar la última secuencia para predecir los próximos 30 días
    input_seq = X_tensor[-seq_len:].unsqueeze(0)  # shape: (1, seq_len, features)
    print(input_seq.shape)
#%%
    #Cargando modelo
    try :
        model=mlflow.pytorch.load_model("I:/Mi unidad/Data Science/Projects/Hotel Booking Forcast & Sentiment Analysis/Model/model/mlruns/334396491036538737/cea8552ba34e4e75aeb49fbe8efe0751/artifacts/model")
       
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise
#%%
    model.eval()
    # Elimina la predicción iterativa, solo pasa la secuencia una vez
    with torch.no_grad():
        pred = model(input_seq)  # input_seq shape: (1, 90, features)
        # Si el output es (1, 30, 1), aplana a (30,)
        if pred.ndim == 3:
            preds = pred[0, :, 0].cpu().numpy()
        elif pred.ndim == 2:
            preds = pred[0].cpu().numpy()
        else:
            preds = pred.cpu().numpy()

    print(preds)
#%%

    # Crear selector de fecha para los próximos 30 días
    future_dates = pd.date_range(df_hotel_final["date"].max() + pd.Timedelta(days=1), periods=30)
    selected_date = st.date_input(
        "Selecciona una fecha para predecir reservas efectivas:",
        min_value=future_dates[0].date(),
        max_value=future_dates[-1].date(),
        value=future_dates[0].date()
    )
    if st.button("Predecir Reserva"):
        # Buscar el índice de la fecha seleccionada
        try:
            idx = (future_dates.date == selected_date).nonzero()[0][0]
            pred_value = preds[idx]
            st.success(f"Predicción de reservas efectivas para {selected_date}: {pred_value:.0f}")
        except IndexError:
            st.error("Fecha seleccionada fuera de rango.")

# Llama a la función principal de la app
if __name__ == "__main__":
    prediction()
