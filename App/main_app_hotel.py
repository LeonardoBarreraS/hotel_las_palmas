import streamlit as st
from analisis_review_app_hotel import display_review_analysis
from prediction_app_hotel import prediction
from Training import *

# Configuración de la página
st.set_page_config(page_title="Hotel Cartagena de Indias", page_icon="🏨", layout="centered")

# Estilos vintage caribeños
st.markdown("""
<style>
    .stApp {
        background-color: #F8ECD1;
    }
    h1 {
        color: #117A65;
        font-family: 'Palatino Linotype','Book Antiqua','Georgia', serif;
    }
    h2 {
        color: #0E6655;
        font-family: 'Palatino Linotype','Book Antiqua','Georgia', serif;
    }
    .sidebar .sidebar-content {
        background-color: #D1F2EB;
    }
    /* Estilo de botones vintage para navegación */
    .stSidebar [role="radiogroup"] > label {
        background-color: #D1F2EB;
        color: #0E6655;
        padding: 8px;
        margin: 5px 0;
        border-radius: 8px;
        font-family: 'Palatino Linotype','Book Antiqua','Georgia', serif;
    }
    .stSidebar [role="radiogroup"] > label:hover {
        background-color: #B2E2DC;
    }
    .stSidebar [role="radiogroup"] input[type="radio"] {
        display: none;
    }
    .stSidebar [role="radiogroup"] input[type="radio"]:checked + div {
        background-color: #D1F2EB;  /* mismo color claro para seleccionado */
        color: #0E6655;
    }
</style>
""", unsafe_allow_html=True)

# Navegación lateral
st.sidebar.title("Navegación")
# Etiquetas con emojis alusivos a cada sección
seccion = st.sidebar.radio("Ir a:", ["🏠 Inicio", "📝 Análisis de Reviews", "📊 Predicción de Reservas"])

if seccion == "🏠 Inicio":
    # Imagen de hotel caribeño en Cartagena
    
    st.title("Bienvenidos al Hotel Las Palmas")
    st.image("images/hotel_cartagena.jpg", width=400, use_container_width=True, caption="Hotel Las Palmas")
    st.markdown("""
    ### Bienvenido a la aplicación de análisis de Reviews y predicción de reservas del Hotel las Palmas en Cartagena de Indias.
    
    **Explora las opiniones de huéspedes y predice el número de reservas para optimizar la planificación.**
                
    **Análisis de Reviews:** Con un set de 5000 reseñas: 
    
    • Se analizan las opiniones de los huéspedes para entender si el hotel cumple con su estrategia de servicio al cliente.
    
    • Del análisis de las opiniones, extraemos sugerencias, agrupamos y las cuantificamos. 
    
    • Cuantificamos la opiniones positivas y negativas, calculamos un score de satisfacción diario y mensual, y usamos el score diario, entre otras variables, para predecir el número de reservas.
    
                
    **Predicción de Reservas:** 
    
    • Teniendo en cuenta 9 variables, incluyendo el número de reservas diario, y con datos de 2 años,  ofrecemos la capacidad de predecir el número de reservas para un día cualquiera en un rango de 30 días al futuro.           
    """)
elif seccion == "📝 Análisis de Reviews":
    st.header("Análisis de Reviews")
    # Llamar a la función para mostrar el análisis de reseñas
    display_review_analysis()


elif seccion == "📊 Predicción de Reservas":
    st.header("Predicción de Reservas")
    prediction()
