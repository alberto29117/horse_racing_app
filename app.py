import streamlit as st
import pandas as pd
import requests
import json
import joblib
import time
from sqlalchemy import create_engine, text

# --- CONFIGURACIÓN INICIAL Y CONEXIONES ---

st.set_page_config(page_title="Análisis Hípico con IA", layout="wide")

# Cargar secretos
try:
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
    RACING_API_KEY = st.secrets["RACING_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    APP_USERNAME = st.secrets["APP_USERNAME"]
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
    
    # Conexión a la base de datos Neon (PostgreSQL)
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

except (FileNotFoundError, KeyError):
    st.error("ERROR: El archivo 'secrets.toml' no está configurado o faltan claves. Por favor, créelo en la carpeta .streamlit.")
    st.stop()

# Cargar el modelo predictivo pre-entrenado
try:
    model_pipeline = joblib.load('lgbm_model.joblib')
except FileNotFoundError:
    st.warning("Advertencia: No se encontró el archivo del modelo 'lgbm_model.joblib'. La funcionalidad de predicción estará deshabilitada. Ejecute 'train_model.py' para crearlo.")
    model_pipeline = None

# --- FUNCIONES AUXILIARES ---

def check_password():
    """Devuelve `True` si el usuario ha iniciado sesión."""
    def login_form():
        with st.form("credentials"):
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.form_submit_button("Iniciar Sesión", on_click=password_entered)

    def password_entered():
        if st.session_state["username"] == APP_USERNAME and st.session_state["password"] == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Usuario o contraseña incorrectos.")
    return False

def fetch_racing_data():
    """Obtiene los datos de las carreras de TheRacingAPI."""
    url = "https://the-racing-api1.p.rapidapi.com/v1/racecards/free"
    headers = {
        "x-rapidapi-key": RACING_API_KEY,
        "x-rapidapi-host": "the-racing-api1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get('racecards', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error al contactar TheRacingAPI: {e}")
        return []

def store_data_in_db(racecards):
    """Guarda los datos de las carreras y corredores en la base de datos."""
    # Aquí iría la lógica para insertar los datos en las tablas 'races' y 'runners'
    # Por simplicidad, en esta demo lo omitimos, pero es un paso crucial en producción.
    st.success(f"{len(racecards)} carreras guardadas en la base de datos (simulado).")

def call_gemini_api(prompt):
    """Llama a la API de Gemini y devuelve la respuesta."""
    # Esta es una implementación simulada. Deberías usar el SDK de google-generativeai
    # import google.generativeai as genai
    # genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel('gemini-1.5-pro-latest')
    # response = model.generate_content(prompt)
    # return response.text
    
    # Simulación para no gastar API calls durante el desarrollo
    time.sleep(2) # Simular latencia de la red
    mock_response = {
        "perfil": {"cuota_betfair": round(np.random.uniform(2.0, 20.0), 2)},
        "analisis_forma": [{"comentario_in_running": "ran on well"}],
        "synergy_analysis": {"swot_balance_score": np.random.randint(-5, 5)}
    }
    return json.dumps(mock_response) # Devolver un JSON como string

def run_ai_analysis(race_data):
    """Itera sobre los corredores y enriquece los datos con análisis de IA."""
    st.info("Iniciando análisis con IA. Este proceso puede tardar varios minutos...")
    
    # Cargar plantillas de prompts
    with open("prompts/prompt_caballos.txt") as f:
        prompt_template_caballo = f.read()
    # ... cargar los otros 3 prompts ...
    
    progress_bar = st.progress(0)
    total_runners = sum(len(race.get('runners', [])) for race in race_data)
    processed_runners = 0
    
    all_runners_data = []

    for race in race_data:
        for runner in race.get('runners', []):
            # Formatear el prompt con datos reales
            prompt_caballo = prompt_template_caballo.format(
                horse_name=runner.get('horse', 'N/A'),
                course=race.get('course', 'N/A'),
                race_date=race.get('date', 'N/A'),
                race_time=race.get('off_time', 'N/A')
            )
            
            # Llamar a la IA (en un caso real, harías 4 llamadas por corredor)
            try:
                ai_response_str = call_gemini_api(prompt_caballo)
                ai_data = json.loads(ai_response_str)
                
                # Enriquecer el runner con los datos de la IA
                runner['ai_analysis'] = ai_data
                runner['cuota_mercado'] = ai_data.get('perfil', {}).get('cuota_betfair', 999.0)
                # Extraer características para el modelo
                runner['swot_balance_score'] = ai_data.get('synergy_analysis', {}).get('swot_balance_score', 0)
                runner['in_running_comment'] = ai_data.get('analisis_forma', [{}])[0].get('comentario_in_running', '')

                all_runners_data.append(runner)

            except Exception as e:
                st.warning(f"No se pudo analizar a {runner.get('horse')}: {e}")

            processed_runners += 1
            progress_bar.progress(processed_runners / total_runners)
            
    st.success("Análisis con IA completado.")
    return all_runners_data


def generate_value_bets(runners_df):
    """Aplica el modelo, identifica valor y genera las apuestas."""
    if model_pipeline is None:
        st.error("El modelo no está cargado. No se pueden generar apuestas.")
        return []
    
    # 1. Aplicar el pipeline de características para obtener predicciones
    features_for_model = runners_df[['official_rating', 'age', 'weight_lbs', 'swot_balance_score', 'course', 'jockey_name', 'trainer_name', 'in_running_comment']]
    
    # El modelo espera un target 'is_winner', que no tenemos para datos futuros.
    # El pipeline se ajustó para manejar esto.
    try:
        probabilities = model_pipeline.predict_proba(features_for_model)
        runners_df['p_modelo'] = probabilities[:, 1] # Probabilidad de la clase '1' (ganador)
    except Exception as e:
        st.error(f"Error al predecir con el modelo: {e}")
        st.info("Asegúrese de que los datos de entrada coinciden con los del entrenamiento.")
        return []

    # 2. Calcular Valor
    runners_df['p_implicita'] = 1 / runners_df['cuota_mercado']
    runners_df['valor'] = (runners_df['p_modelo'] * runners_df['cuota_mercado']) - 1

    # 3. Filtrar apuestas de valor
    value_bets_df = runners_df[runners_df['valor'] > 0.05].sort_values(by='valor', ascending=False)
    
    # 4. Lógica de Staking (simplificada para la demo)
    final_bets = []
    puntos_restantes = 15.0
    
    for _, bet in value_bets_df.iterrows():
        if puntos_restantes <= 0:
            break
        
        # Lógica de Kelly (muy simplificada)
        stake_sugerido = 2.0 # En una app real, usarías la fórmula de Kelly
        
        if puntos_restantes >= stake_sugerido:
            final_bets.append({
                "horse_name": bet['horse'],
                "course": bet['course'],
                "time": bet['off_time'],
                "odds": bet['cuota_mercado'],
                "stake_type": f"{stake_sugerido}PT",
                "stake_points": stake_sugerido
            })
            puntos_restantes -= stake_sugerido

    return final_bets

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🏇 Aplicación de Apuestas de Hípica con IA")

if not check_password():
    st.stop()

st.sidebar.success("Sesión iniciada con éxito.")
st.sidebar.markdown("---")
st.sidebar.header("Flujo de Trabajo")

# Botón para Paso 1
if st.sidebar.button("Paso 1: Obtener Carreras de Mañana"):
    with st.spinner("Obteniendo datos de TheRacingAPI..."):
        race_data = fetch_racing_data()
        if race_data:
            st.session_state.race_data = race_data
            store_data_in_db(race_data) # Simulado
        else:
            st.error("No se pudieron obtener datos de las carreras.")

# Mostrar datos de carreras si existen
if 'race_data' in st.session_state:
    st.subheader("Carreras Obtenidas para Mañana")
    for race in st.session_state.race_data:
        with st.expander(f"📍 {race.get('course', 'N/A')} {race.get('off_time', '')} - {race.get('race_name', 'N/A')}"):
            runners = race.get('runners', [])
            if runners:
                df = pd.DataFrame(runners)
                st.dataframe(df[['horse', 'jockey', 'trainer', 'age', 'sex']])
            else:
                st.write("No hay corredores para esta carrera.")
    
    st.sidebar.markdown("---")
    # Botón para Paso 2 y 3
    if st.sidebar.button("Paso 2 y 3: Analizar y Generar Apuestas"):
        # Ejecutar análisis de IA
        enriched_runners = run_ai_analysis(st.session_state.race_data)
        
        if enriched_runners:
            # Crear DataFrame para el motor de apuestas
            runners_df = pd.DataFrame(enriched_runners)
            # Añadir columnas faltantes para el modelo con valores por defecto
            for col in ['official_rating', 'age', 'weight_lbs']:
                 if col not in runners_df:
                     runners_df[col] = runners_df[col].fillna(0).astype(int)

            # Generar apuestas de valor
            final_bets = generate_value_bets(runners_df)
            st.session_state.final_bets = final_bets
        else:
            st.warning("No se pudieron analizar los corredores.")

# Mostrar resultados finales
if 'final_bets' in st.session_state:
    st.subheader("✅ Apuestas Confirmadas para Mañana")
    if not st.session_state.final_bets:
        st.warning("No se han encontrado apuestas de valor para mañana.")
    else:
        for bet in st.session_state.final_bets:
            col1, col2, col3 = st.columns(3)
            col1.metric("🐴 Caballo", bet['horse_name'])
            col2.metric("📍 Carrera", f"{bet['course']} {bet['time']}")
            col3.metric("📈 Cuota", f"{bet['odds']:.2f}")
            st.info(f"**Apuesta Sugerida:** {bet['stake_type']}")
            st.divider()
