import streamlit as st
import pandas as pd
import requests
import json
import joblib
import time
import io
import numpy as np
from sqlalchemy import create_engine

# --- CONFIGURACIÓN INICIAL Y CONEXIONES ---

st.set_page_config(page_title="Análisis Hípico con IA", layout="wide")

# URL del repositorio de GitHub apuntando al contenido "raw".
# ¡VERSIÓN FINAL CORREGIDA! Apunta a la raíz de tu repositorio.
GITHUB_RAW_URL = "https://raw.githubusercontent.com/alberto29117/horse_racing_app/main/"

# --- CARGA DE RECURSOS DESDE GITHUB (CACHEADO) ---

@st.cache_resource
def load_model_from_github():
    """Carga el modelo .joblib desde GitHub y lo cachea."""
    model_url = f"{GITHUB_RAW_URL}lgbm_model.joblib"
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        model_file = io.BytesIO(response.content)
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo 'lgbm_model.joblib' desde GitHub: {e}")
        return None

@st.cache_data
def load_prompt_from_github(prompt_filename):
    """Carga un archivo de prompt desde GitHub y lo cachea."""
    prompt_url = f"{GITHUB_RAW_URL}prompts/{prompt_filename}"
    try:
        response = requests.get(prompt_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error al cargar el prompt '{prompt_filename}' desde GitHub: {e}")
        return ""

# Cargar el modelo y los prompts al iniciar la app
model_pipeline = load_model_from_github()
PROMPT_TEMPLATES = {
    "caballo": load_prompt_from_github("prompt_caballos.txt"),
    "jockey": load_prompt_from_github("prompt_jockey.txt"),
    "entrenador": load_prompt_from_github("prompt_entrenador.txt"),
    "sinergia": load_prompt_from_github("prompt_sinergia.txt"),
}

# --- CONFIGURACIÓN DE SECRETOS Y BASE DE DATOS (VERSIÓN CORREGIDA FINAL) ---

try:
    # Cargar claves directamente (estructura plana, como en tu secrets.toml)
    RACING_API_KEY = st.secrets["RACING_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    APP_USERNAME = st.secrets["APP_USERNAME"]
    APP_PASSWORD = st.secrets["APP_PASSWORD"]

    # Lógica mejorada para la conexión a la base de datos
    DB_HOST = st.secrets["DB_HOST"]
    if "postgresql://" in DB_HOST:
        # Si el host ya es una URL de conexión completa, la usamos directamente.
        DATABASE_URL = DB_HOST
    else:
        # Si no, la construimos como antes (esto es un respaldo).
        DB_USER = st.secrets["DB_USER"]
        DB_PASSWORD = st.secrets["DB_PASSWORD"]
        DB_PORT = st.secrets["DB_PORT"]
        DB_NAME = st.secrets["DB_NAME"]
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    engine = create_engine(DATABASE_URL)

except (KeyError, FileNotFoundError) as e:
    st.error(f"ERROR: No se pudo cargar una clave desde 'secrets.toml'. Revisa que el archivo exista en la carpeta .streamlit y que todas las claves estén definidas. Clave faltante: {e}")
    st.stop()

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
    st.success(f"{len(racecards)} carreras guardadas en la base de datos (simulado).")

def call_gemini_api(prompt):
    """Llama a la API de Gemini y devuelve la respuesta."""
    # --- IMPLEMENTACIÓN REAL (DESCOMENTAR EN PRODUCCIÓN) ---
    # import google.generativeai as genai
    # try:
    #     genai.configure(api_key=GEMINI_API_KEY)
    #     model = genai.GenerativeModel('gemini-1.5-pro-latest')
    #     response = model.generate_content(prompt)
    #     return response.text
    # except Exception as e:
    #     st.error(f"Error al configurar la API de Gemini. Verifica tu API Key. Error: {e}")
    #     return None
    
    # --- SIMULACIÓN PARA DESARROLLO ---
    time.sleep(1)
    mock_response = {
        "perfil": {"cuota_betfair": round(np.random.uniform(2.0, 20.0), 2)},
        "analisis_forma": [{"comentario_in_running": "ran on well"}],
        "synergy_analysis": {"swot_balance_score": np.random.randint(-5, 5)}
    }
    return json.dumps(mock_response)

def run_ai_analysis(race_data):
    """Itera sobre los corredores, los enriquece con datos de la IA y estandariza los campos."""
    if any(not template for template in PROMPT_TEMPLATES.values()):
        st.error("No se pudo continuar con el análisis porque faltan los prompts. Revisa las URLs en GitHub.")
        return []

    st.info("Iniciando análisis con IA. Este proceso puede tardar varios minutos...")
    
    progress_bar = st.progress(0)
    total_runners = sum(len(race.get('runners', [])) for race in race_data)
    processed_runners = 0
    all_runners_data = []

    for race in race_data:
        current_course = race.get('course', 'Unknown')
        current_off_time = race.get('off_time', 'N/A')

        for runner in race.get('runners', []):
            # Añadir/estandarizar campos ANTES de usarlos
            runner['course'] = current_course
            runner['off_time'] = current_off_time
            runner['jockey_name'] = runner.pop('jockey', 'Unknown')
            runner['trainer_name'] = runner.pop('trainer', 'Unknown')
            
            runner_info = {
                'horse_name': runner.get('horse', 'N/A'),
                'jockey_name': runner.get('jockey_name', 'N/A'),
                'trainer_name': runner.get('trainer_name', 'N/A'),
                'course': runner.get('course', 'N/A'),
                'race_date': race.get('date', 'N/A'),
                'race_time': runner.get('off_time', 'N/A')
            }
            
            prompt_caballo = PROMPT_TEMPLATES["caballo"].format(**runner_info)
            
            ai_response_str = call_gemini_api(prompt_caballo)
            if ai_response_str:
                try:
                    ai_data = json.loads(ai_response_str)
                    runner['ai_analysis'] = ai_data
                    runner['cuota_mercado'] = ai_data.get('perfil', {}).get('cuota_betfair', 999.0)
                    runner['swot_balance_score'] = ai_data.get('synergy_analysis', {}).get('swot_balance_score', 0)
                    runner['in_running_comment'] = ai_data.get('analisis_forma', [{}])[0].get('comentario_in_running', '')
                    
                    # Añadir datos simulados para el modelo
                    runner['official_rating'] = runner.get('official_rating', np.random.randint(70, 100))
                    runner['weight_lbs'] = runner.get('weight_lbs', np.random.randint(120, 140))
                    runner['age'] = pd.to_numeric(runner.get('age'), errors='coerce')

                    all_runners_data.append(runner)
                except json.JSONDecodeError:
                    st.warning(f"La respuesta de la IA para {runner.get('horse')} no es un JSON válido.")

            processed_runners += 1
            progress_bar.progress(processed_runners / total_runners)
            
    st.success("Análisis con IA completado.")
    return all_runners_data

def generate_value_bets(runners_df):
    """Aplica el modelo, identifica valor y genera las apuestas."""
    if model_pipeline is None:
        st.error("El modelo no está cargado. No se pueden generar apuestas.")
        return []
    
    # Asegurar que todas las columnas necesarias para el modelo existen
    required_features = ['official_rating', 'age', 'weight_lbs', 'swot_balance_score', 'course', 'jockey_name', 'trainer_name', 'in_running_comment']
    for feature in required_features:
        if feature not in runners_df.columns:
            st.error(f"Error Interno: La columna '{feature}' es necesaria para la predicción y no se encontró.")
            return []

    features_for_model = runners_df[required_features]
    
    try:
        probabilities = model_pipeline.predict_proba(features_for_model)
        runners_df['p_modelo'] = probabilities[:, 1]
    except Exception as e:
        st.error(f"Error al predecir con el modelo: {e}")
        return []

    runners_df['p_implicita'] = 1 / runners_df['cuota_mercado']
    runners_df['valor'] = (runners_df['p_modelo'] * runners_df['cuota_mercado']) - 1
    value_bets_df = runners_df[runners_df['valor'] > 0.05].sort_values(by='valor', ascending=False)
    
    final_bets = []
    puntos_restantes = 15.0
    
    for _, bet in value_bets_df.iterrows():
        if puntos_restantes <= 0: break
        stake_sugerido = 2.0
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

if st.sidebar.button("Paso 1: Obtener Carreras de Mañana"):
    with st.spinner("Obteniendo datos de TheRacingAPI..."):
        race_data = fetch_racing_data()
        if race_data:
            st.session_state.race_data = race_data
            # Limpiar datos antiguos de análisis si se obtienen nuevas carreras
            if 'enriched_runners_df' in st.session_state:
                del st.session_state.enriched_runners_df
            store_data_in_db(race_data)
        else:
            st.error("No se pudieron obtener datos de las carreras.")

if 'race_data' in st.session_state:
    st.subheader("Carreras Obtenidas para Mañana")
    
    # Determinar qué datos mostrar: los enriquecidos o los básicos
    if 'enriched_runners_df' in st.session_state:
        # Mostrar datos después del análisis, incluyendo la cuota de la IA
        display_df = st.session_state.enriched_runners_df.copy()
        display_df.rename(columns={'cuota_mercado': 'Cuota (IA)'}, inplace=True)
        cols_to_show = ['horse', 'jockey_name', 'trainer_name', 'age', 'sex', 'Cuota (IA)']
        
        for race_id, group in display_df.groupby(['course', 'off_time', 'race_name']):
            course, off_time, race_name = race_id
            with st.expander(f"📍 {course} {off_time} - {race_name}"):
                st.dataframe(group[cols_to_show])
    else:
        # Mostrar datos básicos antes del análisis
        for race in st.session_state.race_data:
            with st.expander(f"📍 {race.get('course', 'N/A')} {race.get('off_time', '')} - {race.get('race_name', 'N/A')}"):
                runners = race.get('runners', [])
                if runners:
                    df = pd.DataFrame(runners)
                    df.rename(columns={'jockey': 'jockey_name', 'trainer': 'trainer_name'}, inplace=True)
                    display_cols = ['horse', 'jockey_name', 'trainer_name', 'age', 'sex']
                    cols_to_show = [col for col in display_cols if col in df.columns]
                    st.dataframe(df[cols_to_show])
                else:
                    st.write("No hay corredores para esta carrera.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Paso 2 y 3: Analizar y Generar Apuestas"):
        enriched_runners = run_ai_analysis(st.session_state.race_data)
        if enriched_runners:
            # Crear y almacenar el DataFrame enriquecido en la sesión
            runners_df = pd.DataFrame(enriched_runners)
            runners_df.fillna({
                'official_rating': 0, 'age': 0, 'weight_lbs': 0, 
                'swot_balance_score': 0, 'course': 'Unknown', 
                'jockey_name': 'Unknown', 'trainer_name': 'Unknown', 'in_running_comment': ''
            }, inplace=True)
            
            # Añadir race_name para poder agrupar después
            race_name_map = { (r['course'], r['off_time']): r['race_name'] for r in st.session_state.race_data }
            runners_df['race_name'] = runners_df.apply(lambda row: race_name_map.get((row['course'], row['off_time']), 'N/A'), axis=1)

            st.session_state.enriched_runners_df = runners_df
            
            final_bets = generate_value_bets(runners_df)
            st.session_state.final_bets = final_bets
            # Forzar un refresco de la página para mostrar las nuevas tablas
            st.rerun()
        else:
            st.warning("No se pudieron analizar los corredores.")

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
