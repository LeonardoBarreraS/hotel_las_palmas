#%%
import sys
import os
sys.path.append(os.path.abspath(".."))

from utils_pack import *
import pandas as pd
import streamlit as st
import ast
import matplotlib.pyplot as plt
#%%

def analyze_reviews():
    # Load the dataset
    df_reviews = pd.read_csv("../Data/sentiment_analysis_suggestions_themes.csv")
    # Asegúrate de que la columna 'insights' es un diccionario (o conviértela desde string si es necesario)
    if df_reviews["insights"].dtype == object:
        df_reviews["insights"] = df_reviews["insights"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_reviews["sentiment_openai"] = df_reviews["insights"].apply(lambda d: d.get("sentiment") if isinstance(d, dict) else None)
    
 
    # Calculate statistics
    no_reviews = len(df_reviews)
    no_reviews_positive = len(df_reviews[(df_reviews["sentiment"] == "POSITIVE") & (df_reviews["sentiment_openai"] == "positive")])
    percentage_positive = (no_reviews_positive / no_reviews) * 100
    no_reviews_negative = len(df_reviews[(df_reviews["sentiment"] == "NEGATIVE") & (df_reviews["sentiment_openai"] == "negative")])
    percentage_negative = (no_reviews_negative / no_reviews) * 100

    # Return the statistics
    return {
        "no_reviews": no_reviews,
        "no_reviews_positive": no_reviews_positive,
        "percentage_positive": percentage_positive,
        "no_reviews_negative": no_reviews_negative,
        "percentage_negative": percentage_negative
    }, df_reviews
#%%
def display_review_analysis():
    # Ejecutar la función de análisis de reseñas
    stats, df_reviews = analyze_reviews()
    # Visualizar 50 muestras aleatorias de la columna "review"
    # Ajustar el ancho de la página para mejor visualización
    # Ajustar el ancho de la página y del dataframe para mejor visualización
    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 1200px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .stDataFrame, .stTable {
                width: 100% !important;
                min-width: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Línea horizontal debajo del título principal
    st.markdown("<hr style='margin-top: 0; margin-bottom: 25px;'>", unsafe_allow_html=True)

    filtered_reviews = df_reviews[((df_reviews["sentiment"] == "POSITIVE") & (df_reviews["sentiment_openai"] == "positive")) | ((df_reviews["sentiment"] == "NEGATIVE") & (df_reviews["sentiment_openai"] == "negative"))]
    sample_reviews= filtered_reviews.sample(n=50, random_state=42)[["review"]] 
    

    st.markdown("<h3 style='color: #2C3E50; margin-bottom: 10px;'>1. Análisis general</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # Add a text box for describing the section using st.markdown with custom CSS styling
    st.markdown(
        """
        <div style='background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;'>
            <p style='margin: 0; color: #2C3E50; font-size: 1em;'>
             En esta sección podrás visualizar una muestra aleatoria de 50 reviews de un set de datos de 5000 reviews. A través de la AI, podrás clasificarlos como POSITIVOS o NEGATIVOS, y ver estadísticas generales de esta clasificación
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.write("")
    st.write("")


    with st.expander("Mostrar muestra aleatoria de 50 reviews SIN CLAFICAR"):
        st.write("### Muestra aleatoria de 50 reviews")
        st.dataframe(sample_reviews, use_container_width=True)
    
    # Use session state to persist classified reviews and stats
    if "show_classified" not in st.session_state:
        st.session_state["show_classified"] = False
    if "classified_reviews" not in st.session_state:
        st.session_state["classified_reviews"] = None
    if "stats" not in st.session_state:
        st.session_state["stats"] = None

    if st.button("Clasificar Reviews"):
        st.session_state["show_classified"] = True
        st.session_state["classified_reviews"] = filtered_reviews.sample(n=50, random_state=42)[["review", "sentiment"]]
        st.session_state["stats"] = stats

    if st.session_state.get("show_classified", False):
        classified_reviews = st.session_state["classified_reviews"]
        stats = st.session_state["stats"]
        with st.expander("Mostrar muestra aleatoria de 50 reviews CLASIFICADAS"):
            st.write("### Muestra clasificada de 50 reviews")
            st.dataframe(classified_reviews, use_container_width=True)
        # Mostrar estadísticas en una franja horizontal con cuadros para cada indicador
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; margin-top: 20px; margin-bottom: 20px;">
                <div style="flex:1; background-color: #F9E79F; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px #f7dc6f;">
                    <h2 style="color: #B9770E; margin-bottom: 10px;">{stats['no_reviews']}</h2>
                    <p style="font-weight: bold;">Total de Reviews</p>
                </div>
                <div style="flex:1; background-color: #D4EFDF; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #229954; margin-bottom: 5px;">{stats['no_reviews_positive']}</h3>
                    <p>Reviews Positivos</p>
                    <span style="color: #229954; font-size: 1.1em;">{stats['percentage_positive']:.2f}%</span>
                </div>
                <div style="flex:1; background-color: #FADBD8; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #CB4335; margin-bottom: 5px;">{stats['no_reviews_negative']}</h3>
                    <p>Reviews Negativos</p>
                    <span style="color: #CB4335; font-size: 1.1em;">{stats['percentage_negative']:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<h3 style='color: #34495E; margin-top: 30px; margin-bottom: 15px; text-align: left;'>2. Análisis de Impacto en variables estratégicas</h3>",
            unsafe_allow_html=True
        )
        st.markdown("<hr style='margin-top: 0; margin-bottom: 25px;'>", unsafe_allow_html=True)

        st.markdown(
        """
        <div style='background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;'>
            <p style='margin: 0; color: #2C3E50; font-size: 1em;'>
             En esta sección podrás visualizar las variables estratégicas definidas por el hotel, y como a través de la IA se puede vincular cada review negativo con alguna de estas variables.
             <br>
             <br>Podrás ver el porcentaje de reviews negativos relacionados a cada una de las variables estratégicas definidas por el hotel.
             <br>
             <br>Esto te permitirá entender cuáles son las variables que más afectan la experiencia del cliente!!
            </p>
        </div>
        """, 
        unsafe_allow_html=True
        )
        st.write("")
        st.write("")

#%%
        df_variables=pd.read_csv("../Data/sentiment_dataset_huggingface_hotel2.csv")
        st.markdown("<h4 style='color: #2C3E50; margin-bottom: 10px;'>Variables Estratégicas definidas por el Hotel</h4>", unsafe_allow_html=True)
        # Get unique, non-null, non-'na' variables
        unique_vars = [v for v in df_variables["matched_variable_id"].dropna().unique() if str(v).lower() != "na"]

        cleaned_vars_spanish = {
            'staff_friendliness': 'Amabilidad del Personal',
            'room_size': 'Tamaño de la Habitación', 
            'location_accessibility': 'Accesibilidad del Hotel', 
            'room_cleanliness': 'Limpieza de la Habitación', 
            'amenity_quality': 'Calidad de las Instalaciones', 
            'price_quality_ratio': 'Relación Precio-Calidad', 
            'service_personalization': 'Personalización del Servicio', 
            'technological_facilities': 'Instalaciones Tecnológicas'
        }
        # Create spanish_desc column by mapping matched_variable_id to spanish descriptions
        df_variables['spanish_desc'] = df_variables['matched_variable_id'].map(cleaned_vars_spanish)
        strategic_vars=[v for v in df_variables["spanish_desc"].value_counts().index]

                
#%%
        # Display each variable in a styled badge/box
        if strategic_vars:
            badges = "".join(
                f"<span style='display:inline-block; background:#D6EAF8; color:#154360; padding:8px 18px; margin:6px 8px 6px 0; border-radius:18px; font-size:1em; font-weight:500; box-shadow:0 1px 4px #aed6f1;'>{v}</span>"
                for v in strategic_vars
            )
            st.markdown(f"<div style='margin-bottom:18px;'>{badges}</div>", unsafe_allow_html=True)
        else:
            st.info("No strategic variables found.")

        
        st.markdown(
            "<h5 style='color: #34495E; margin-top: 25px; margin-bottom: 10px;'>¿Que % de los reviews negativos están relacionados a cada una de las variables estratégicas del Hotel?</h5>",
            unsafe_allow_html=True
        )
        
        # Calculate total sum of matched_variable_id counts
        total_var_counts = df_variables['spanish_desc'].value_counts().sum()
        
        # Calculate percentages relative to total counts
        var_percentages = (df_variables['spanish_desc'].value_counts() / total_var_counts * 100).sort_values()

        # Create horizontal bar chart
        import plotly.express as px

        fig = px.bar(
            x=var_percentages.values,
            y=var_percentages.index,
            orientation='h',
            labels={'x': 'Porcentaje (%)', 'y': 'Variable Estratégica'},
            text=[f"{v:.1f}%" for v in var_percentages.values],
            color=var_percentages.values,
            color_continuous_scale=px.colors.sequential.Blues,
            title="Distribución de Variables Estratégicas en Reviews"
        )
        fig.update_traces(
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )
        fig.update_layout(
            xaxis_title="Porcentaje (%)",
            yaxis_title="Variable Estratégica",
            title={
            "text": "Distribución de Variables Estratégicas en Reviews",
            "x": 0.5,
            "xanchor": "center"
            },
            plot_bgcolor="#F8F9F9",
            paper_bgcolor="#F8F9F9",
            font=dict(size=14, color="#2C3E50"),
            margin=dict(l=40, r=20, t=60, b=40),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)




        st.markdown(
            "<h3 style='color: #34495E; margin-top: 30px; margin-bottom: 15px; text-align: left;'>3. Extracción y Análisis de Sugerencias de los Reviews</h3>",
            unsafe_allow_html=True
        )
        st.markdown("<hr style='margin-top: 0; margin-bottom: 25px;'>", unsafe_allow_html=True)

        st.markdown(
        """
        <div style='background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;'>
            <p style='margin: 0; color: #2C3E50; font-size: 1em;'>
            En esta sección podrás ver como a través de la IA se puede interpretar el sentimiento de los reviews, y extraer sugerencias de los mismos.
            <br>
            <br>Podrás ver como a partir de una opinión se puede extraer una sugerencia, categorizarla y cuantificar.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
        )
        st.write("")
        st.write("")

        # Use session state to persist suggestion extraction
        if "show_suggestions" not in st.session_state:
            st.session_state["show_suggestions"] = False
        if st.button("Extraer y Análizar sugerencias de los Reviews"):
            st.session_state["show_suggestions"] = True

        if st.session_state.get("show_suggestions", False):
            # Contar la frecuencia de cada categoría en 'desc_tema', excluyendo '-1'
            category_counts = (
                filtered_reviews[filtered_reviews["desc_tema"] != "-1"]["desc_tema"]
                .value_counts()
                .to_dict()
            )
            # Generar un DataFrame con una muestra de 5 reviews por cada categoría de "desc_tema" (sin incluir "-1"),
            # seleccionando los de mayor "valor_tema" e incluyendo la columna "suggestion"
            sample_reviews_by_category = (
                filtered_reviews[filtered_reviews["desc_tema"] != "-1"]
                .sort_values(["desc_tema", "valor_tema"], ascending=[True, False])
                .groupby("desc_tema")
                .head(5)
                .reset_index(drop=True)
            )
            sample_reviews_by_category = sample_reviews_by_category[["desc_tema", "review", "suggestions"]]
            
            
            # Mostrar las categorías principales en una grilla de cuadros tipo textbox pequeños y estilizados (7 columnas)
            st.markdown(
                "<h4 style='margin-bottom: 10px;'>Categorías principales de sugerencias detectadas:</h4>",
                unsafe_allow_html=True
            )

            # Mostrar el indicador "Total de Sugerencias detectadas"
            total_category_count = sum(category_counts.values())
            st.markdown(
                f"""
                <div style="background-color: #D6EAF8; padding: 12px 0 10px 0; border-radius: 8px; text-align: center; width: 220px; margin: 0 auto 15px auto; box-shadow: 0 2px 8px #aed6f1;">
                    <h3 style="color: #2471A3; margin-bottom: 4px; font-size: 1.6em;">{total_category_count}</h3>
                    <p style="font-weight: 500; margin-bottom: 0; font-size: 1em;">Total de sugerencias detectadas</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        

            # Calcular el porcentaje de cada categoría respecto al total de ocurrencias de categorías (no al total de reviews)
            category_percentages = {cat: (count / total_category_count) * 100 for cat, count in category_counts.items()}
            top_categories_percent = sorted(category_percentages.items(), key=lambda x: x[1], reverse=True)[:7]

            # Preparar datos para el gráfico
            categories = [cat for cat, _ in top_categories_percent]
            percentages = [perc for _, perc in top_categories_percent]

            import plotly.express as px

            fig = px.bar(
                x=categories,
                y=percentages,
                labels={"x": "Categoría", "y": "Porcentaje (%)"},
                text=[f"{p:.1f}%" for p in percentages],
                color=percentages,
                color_continuous_scale=px.colors.sequential.Blues,
                title="Porcentaje de Reviews por Categoría de Sugerencia"
            )
            fig.update_traces(
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            fig.update_layout(
                xaxis_title="Categoría",
                yaxis_title="Porcentaje (%)",
                title={
                    "text": "Porcentaje de Reviews por Categoría de Sugerencia",
                    "x": 0.5,
                    "xanchor": "center"
                },  # Centra el título
                plot_bgcolor="#F8F9F9",
                paper_bgcolor="#F8F9F9",
                font=dict(size=14, color="#2C3E50"),
                margin=dict(l=40, r=20, t=60, b=40),
                height=510,  # Reducido ~15% desde 600
                yaxis=dict(range=[0, 100])  # Escala del eje y hasta 100%
            )
            st.plotly_chart(fig, use_container_width=True)


            # Obtener las 7 categorías más frecuentes
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:7]

            # Construir la tabla HTML con texto más pequeño y mayor ancho
            table_html = "<table style='width:100%; table-layout:fixed;'><tr>"
            for cat, count in top_categories:
                table_html += (
                    f"<td style='background:#F4F6F6; border-radius:8px; padding:10px; text-align:center; font-size:0.85em; border:1px solid #D5D8DC; word-break:break-word; min-width:220px; max-width:270px;'>"
                    f"<b style='font-size:0.95em;'>{cat}</b><br>"
                    f"<span style='color:#5D6D7E; font-size:0.85em;'>{count} veces</span>"
                    f"</td>"
                )
            table_html += "</tr></table>"

            st.markdown(table_html, unsafe_allow_html=True)

            # Añadir Título 4: Cuantificación del Sentimiento
            st.markdown(
                "<h3 style='color: #34495E; margin-top: 30px; margin-bottom: 15px; text-align: left;'>4. Cuantificación del Sentimiento</h3>",
                unsafe_allow_html=True
            )
            st.markdown("<hr style='margin-top: 0; margin-bottom: 25px;'>", unsafe_allow_html=True)


            st.markdown(
            """
            <div style='background-color: #EBF5FB; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;'>
                <p style='margin: 0; color: #2C3E50; font-size: 1em;'>
                Cada opinión la cuantificamos. Si es Negativa (-1) o Positiva (+1). Sacamos el promedio de cada día y el promedio de cada mes. En la gráfica podrás ver el promedio de cada mes. 
                Entre más cercano a 1, más positivo es el sentimiento. Entre más cercano a -1, más negativo es el sentimiento.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
            )
            st.write("")
            st.write("")

            senti=pd.read_csv("../Data/sentiment_huggingface_avg_daily_sentiment.csv")
            #converting date to datetime
            senti["date"]=pd.to_datetime(senti["date"])
            #creating a date with year and month
            senti["year_month"]=senti["date"].dt.to_period("M").dt.to_timestamp()
            #grouping by year and month
            senti_year_month=senti.groupby("year_month").agg({"sentiment_numeric":"mean"}).reset_index()

            import plotly.express as px

            fig = px.bar(
                senti_year_month,
                x="year_month",
                y="sentiment_numeric",
                labels={"year_month": "Mes", "sentiment_numeric": "Sentimiento Promedio"},
                title="Sentimiento Promedio por Mes de Reserva",
                color="sentiment_numeric",
                color_continuous_scale=px.colors.sequential.Blues,
                template="plotly_white"
            )
            fig.update_layout(
                xaxis_title="Mes de Reserva",
                yaxis_title="Sentimiento Cuantificado",
                title_x=0.5,
                bargap=0.2,
                plot_bgcolor="#F8F9F9"
            )
            fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
            st.plotly_chart(fig, use_container_width=True)




# %%
