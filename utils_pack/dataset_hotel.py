#%%
import pandas as pd
import holidays
import numpy as np
from datetime import datetime, timedelta

#%%
#Calcular fechas dinámicamente
end_date=(datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')
start_date=(datetime.today() - timedelta(days=733)).strftime('%Y-%m-%d')


#Temperature data creation
url = f"https://archive-api.open-meteo.com/v1/archive?latitude=10.3910&longitude=-75.4794&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&timezone=America%2FBogota&format=csv"


def generate_dataframe():

    try:

        #%%
        # Leer el archivo CSV a partir de la cuarta fila
        df_temp = pd.read_csv(url, skiprows=3)

        # Renombrar la columna 'time0' a 'date'
        df_temp.rename(columns={'time': 'date'}, inplace=True)

        df_temp["date"] = pd.to_datetime(df_temp["date"])
       
#%%
        #Holiday data creation

        fechas= pd.date_range(start=start_date, end=end_date)

        years = list(set(fechas.year))  # Obtener los años únicos del rango de fechas
        feriados_col = holidays.country_holidays('CO', observed=True, years=years)

        df_holidays= pd.DataFrame({"dates": fechas})
        df_holidays["is_holiday"]= df_holidays["dates"].isin(feriados_col)
        df_holidays["is_weekend"]= df_holidays["dates"].dt.dayofweek >= 5
        df_holidays["day_of_week"]= df_holidays["dates"].dt.day_of_week
        df_holidays["month"]= df_holidays["dates"].dt.month
        df_holidays["week_of_year"]= df_holidays["dates"].dt.isocalendar().week


        #%%
        #Joining dataframes

        df_merged=df_holidays.merge(df_temp, left_on="dates", right_on="date", how="left")


        df_merged.drop(columns=["date"], inplace=True)
        df_merged.rename(columns={"dates": "date", "temperature_2m_mean (°C)":"temperature"}, inplace=True)

        # %%
        #Final Dataframe

        df_dates_temp=df_merged.drop(columns=["temperature_2m_max (°C)", "temperature_2m_min (°C)"])

        df=df_dates_temp.copy()

        #%% 
        #Promotions_days

        df['promotion'] = df.apply(
            lambda row: np.random.rand() < 0.4 if row['is_holiday'] or row['is_weekend'] else np.random.rand() < 0.1,
            axis=1
        )

        #Local_event
        df['local_event'] = np.random.rand(len(df)) < 0.07 # 7% chance of local event

        #Avg. room Price

        base_price = 150000  # en pesos colombianos
        price_variation = []

        for idx, row in df.iterrows():
            price = base_price
            if row['is_weekend']:
                price *= 1.1
            if row['is_holiday']:
                price *= 1.2
            if row['local_event']:
                price *= 1.15
            noise = np.random.normal(0, 10000)  # variación aleatoria
            price_variation.append(int(max(price + noise, 150000)))  # mínimo 150.000

        df['avg_room_price'] = price_variation


        #%% 
        #Bookings

        month_base = {
            1: 45,   # Enero (temporada alta)
            2: 35,
            3: 30,
            4: 32,
            5: 28,
            6: 40,   # Mitad de año (vacaciones escolares)
            7: 42,
            8: 33,
            9: 25,
            10: 27,
            11: 30,
            12: 50   # Navidad/Año nuevo
        }


        def generate_bookings(row):
            base = month_base[row['month']]
            # Ajuste por temperatura (óptima entre 27-30 °C)
            base += (row['temperature'] - 27) * 1.2

            # Fin de semana
            if row['is_weekend']:
                base += 4

            # Feriado
            if row['is_holiday']:
                base += 6

            # Evento local
            if row['local_event']:
                base += 8

            # Promoción activa
            if row['promotion']:
                base += 5

            # Variabilidad aleatoria
            noise = np.random.normal(0, 5)

            # Reservas finales (mínimo 0, máximo 60 habitaciones)
            bookings = max(0, min(round(base + noise), 60))
            return bookings

        df['bookings'] = df.apply(generate_bookings, axis=1)



        # %%
        #Occupancy rate

        total_rooms = 60
        df['occupancy_rate_prev_day'] = df['bookings'].shift(1) / total_rooms
        df['occupancy_rate_prev_day'] = df['occupancy_rate_prev_day'].fillna(0)
        # %%
        #Cancellation rate

        def simulate_cancellation(row):
            rate = 0.10  # tasa base del 10%
            if row['promotion']:
                rate += 0.05
            if row['local_event']:
                rate -= 0.03
            if row['is_holiday']:
                rate -= 0.02
            rate += np.random.normal(0, 0.01)  # ruido aleatorio
            return min(max(rate, 0), 0.5)  # limitar entre 0 y 50%

        df['cancellation_rate'] = df.apply(simulate_cancellation, axis=1)

        #Effective bookings
        df['effective_bookings'] = (df['bookings'] * (1 - df['cancellation_rate'])).astype(int)
        return df
    except Exception as e:
        print(f"Error en generate_dataframe: {e}")
        return None
#%%
if __name__=="__main__":
    df=generate_dataframe()
