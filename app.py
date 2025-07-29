import streamlit as st
import pandas as pd
import requests
import json
import joblib
import time
import io
import numpy as np
from sqlalchemy import create_engine
import google.generativeai as genai
from datetime import date, timedelta

# --- CONFIGURACIÓN INICIAL Y CONEXIONES ---

st.set_page_config(page_title="Análisis Hípico con IA", layout="wide")

# URL del repositorio de GitHub apuntando al contenido "raw".
GITHUB_RAW_URL = "https://raw.githubusercontent.com/alberto29117/horse_racing_app/main/"

# --- INICIALIZACIÓN DEL ESTADO DE LA APLICACIÓN ---
# Usamos st.session_state para mantener los datos entre interacciones.
# Esto actuará como nuestra base de datos temporal.

if 'historical_bets' not in st.session_state:
    # DataFrame para almacenar todas las apuestas guardadas
    st.session_state.historical_bets = pd.DataFrame(columns=[
        'bet_id', 'horse_name', 'course', 'time', 'stake_type', 
        'stake_points', 'ia_odds', 'placed_odds', 'status', 'pnl'
    ])

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

# --- CONFIGURACIÓN DE SECRETOS Y BASE DE DATOS ---

try:
    RACING_API_KEY = st.secrets["RACING_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    APP_USERNAME = st.secrets["APP_USERNAME"]
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
    DB_HOST = st.secrets["DB_HOST"]
    if "postgresql://" in DB_HOST:
        DATABASE_URL = DB_HOST
    else:
        DB_USER = st.secrets["DB_USER"]
        DB_PASSWORD = st.secrets["DB_PASSWORD"]
        DB_PORT = st.secrets["DB_PORT"]
        DB_NAME = st.secrets["DB_NAME"]
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
except (KeyError, FileNotFoundError) as e:
    st.error(f"ERROR: No se pudo cargar una clave desde 'secrets.toml'. Revisa la configuración. Error: {e}")
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
        if st.session_state.get("username") == APP_USERNAME and st.session_state.get("password") == APP_PASSWORD:
            st.session_state["password_correct"] = True
            if "password" in st.session_state: del st.session_state["password"]
            if "username" in st.session_state: del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Usuario o contraseña incorrectos.")
    return False

def fetch_racing_data():
    """Obtiene los datos de las carreras de TheRacingAPI para el día siguiente."""
    # Calcular la fecha de mañana
    tomorrow = date.today() + timedelta(days=1)
    date_str = tomorrow.strftime('%Y-%m-%d')

    # Usamos el endpoint que permite especificar una fecha.
    # Nota: El plan gratuito de la API podría no permitir el acceso a fechas futuras.
    url = "https://the-racing-api1.p.rapidapi.com/v1/racecards"
    params = {"date": date_str}
    
    headers = {"x-rapidapi-key": RACING_API_KEY, "x-rapidapi-host": "the-racing-api1.p.rapidapi.com"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('racecards', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error al contactar TheRacingAPI: {e}")
        # Añadir un mensaje más específico si es un error de cliente (403/401)
        if e.response and e.response.status_code in [401, 403]:
            st.warning("Este error puede indicar que tu plan de API no permite acceder a fechas futuras. El endpoint gratuito suele estar limitado al día de hoy.")
        return []

def call_gemini_api(prompt):
    """Llama a la API de Gemini y devuelve la respuesta en formato JSON."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {"response_mime_type": "application/json"}
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
             st.error(f"Límite de cuota de la API de Gemini alcanzado. El proceso se reanudará en breve. Error: {e}")
        else:
            st.error(f"Error al contactar la API de Gemini. Verifica tu API Key. Error: {e}")
        return "{}"

def run_ai_analysis(race_data):
    """Itera sobre los corredores, los enriquece con datos de la IA y estandariza los campos."""
    st.info("Iniciando análisis con IA. Este proceso puede tardar varios minutos...")
    progress_bar = st.progress(0)
    total_runners = sum(len(race.get('runners', [])) for race in race_data)
    processed_runners = 0
    all_runners_data = []

    for race in race_data:
        for runner in race.get('runners', []):
            runner['course'] = race.get('course', 'Unknown')
            runner['off_time'] = race.get('off_time', 'N/A')
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
                    runner['cuota_mercado'] = ai_data.get('perfil', {}).get('cuota_betfair', 999.0)
                    runner['swot_balance_score'] = ai_data.get('synergy_analysis', {}).get('swot_balance_score', 0)
                    
                    # CORRECCIÓN: Manejo seguro de la lista 'analisis_forma'
                    analisis_forma_list = ai_data.get('analisis_forma', [])
                    if analisis_forma_list:
                        runner['in_running_comment'] = analisis_forma_list[0].get('comentario_in_running', '')
                    else:
                        runner['in_running_comment'] = ''

                    runner['official_rating'] = runner.get('official_rating', np.random.randint(70, 100))
                    runner['weight_lbs'] = runner.get('weight_lbs', np.random.randint(120, 140))
                    runner['age'] = pd.to_numeric(runner.get('age'), errors='coerce')
                    all_runners_data.append(runner)
                except json.JSONDecodeError:
                    st.warning(f"La respuesta de la IA para {runner.get('horse')} no es un JSON válido.")
            
            processed_runners += 1
            progress_bar.progress(processed_runners / total_runners)
            
            time.sleep(3) 
            
    st.success("Análisis con IA completado.")
    return all_runners_data

def generate_value_bets(runners_df):
    """Aplica el modelo, identifica valor y genera las apuestas."""
    if model_pipeline is None: return []
    required_features = ['official_rating', 'age', 'weight_lbs', 'swot_balance_score', 'course', 'jockey_name', 'trainer_name', 'in_running_comment']
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
                "horse_name": bet['horse'], "course": bet['course'],
                "time": bet['off_time'], "ia_odds": bet['cuota_mercado'],
                "stake_type": f"{stake_sugerido}PT", "stake_points": stake_sugerido
            })
            puntos_restantes -= stake_sugerido
    return final_bets

def update_bet_status(bet_id):
    """Actualiza el estado de una apuesta y calcula su P/L."""
    new_status = st.session_state[f"status_{bet_id}"]
    idx = st.session_state.historical_bets.index[st.session_state.historical_bets['bet_id'] == bet_id][0]
    
    st.session_state.historical_bets.loc[idx, 'status'] = new_status
    
    bet = st.session_state.historical_bets.loc[idx]
    stake = bet['stake_points']
    odds = bet['placed_odds']
    pnl = 0.0

    if new_status == "Ganada":
        pnl = (stake * odds) - stake
    elif new_status == "Perdida":
        pnl = -stake
    elif new_status == "Colocada":
        if "ew" in bet['stake_type'].lower():
            place_odds = 1 + ((odds - 1) / 5)
            pnl = (stake / 2 * place_odds) - stake
        else:
            pnl = -stake
    elif new_status == "No Corrió":
        pnl = 0.0

    st.session_state.historical_bets.loc[idx, 'pnl'] = pnl

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🏇 Aplicación de Apuestas de Hípica con IA")

if not check_password():
    st.stop()

st.sidebar.success("Sesión iniciada con éxito.")
st.sidebar.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Análisis Diario", "Gestionar Apuestas Pendientes", "Historial de Resultados", "Rendimiento General"])

# --- PESTAÑA 1: ANÁLISIS DIARIO ---
with tab1:
    st.header("Flujo de Trabajo Diario")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Paso 1: Obtener Carreras de Mañana", use_container_width=True):
            with st.spinner("Obteniendo datos de TheRacingAPI..."):
                st.session_state.race_data = fetch_racing_data()
                if 'final_bets' in st.session_state: del st.session_state.final_bets
                if 'enriched_runners_df' in st.session_state: del st.session_state.enriched_runners_df
    
    with col_b:
        is_disabled = 'race_data' not in st.session_state
        if st.button("Paso 2: Analizar y Generar Apuestas", use_container_width=True, disabled=is_disabled):
            with st.spinner("Analizando con IA y generando predicciones..."):
                enriched_runners = run_ai_analysis(st.session_state.race_data)
                if enriched_runners:
                    runners_df = pd.DataFrame(enriched_runners)
                    runners_df.fillna(0, inplace=True)
                    
                    race_name_map = { (r['course'], r['off_time']): r.get('race_name', 'N/A') for r in st.session_state.race_data }
                    runners_df['race_name'] = runners_df.apply(lambda row: race_name_map.get((row['course'], row['off_time']), 'N/A'), axis=1)

                    st.session_state.enriched_runners_df = runners_df
                    st.session_state.final_bets = generate_value_bets(runners_df)
                    st.rerun()
                else:
                    st.warning("No se pudieron analizar los corredores.")

    st.divider()

    if 'race_data' in st.session_state:
        st.subheader("Listado de Carreras")
        
        if 'enriched_runners_df' in st.session_state:
            display_df = st.session_state.enriched_runners_df.copy()
            display_df.rename(columns={'cuota_mercado': 'Cuota (IA)'}, inplace=True)
            cols_to_show = ['horse', 'jockey_name', 'trainer_name', 'age', 'sex', 'Cuota (IA)']
            
            for race_id, group in display_df.groupby(['course', 'off_time', 'race_name']):
                course, off_time, race_name = race_id
                with st.expander(f"📍 {course} {off_time} - {race_name}", expanded=True):
                    existing_cols = [col for col in cols_to_show if col in group.columns]
                    st.dataframe(group[existing_cols])
        else:
            for race in st.session_state.race_data:
                with st.expander(f"📍 {race.get('course', 'N/A')} {race.get('off_time', '')} - {race.get('race_name', 'N/A')}", expanded=True):
                    runners = race.get('runners', [])
                    if runners:
                        df = pd.DataFrame(runners)
                        df.rename(columns={'jockey': 'jockey_name', 'trainer': 'trainer_name'}, inplace=True)
                        display_cols = ['horse', 'jockey_name', 'trainer_name', 'age', 'sex']
                        cols_to_show = [col for col in display_cols if col in df.columns]
                        st.dataframe(df[cols_to_show])
                    else:
                        st.write("No hay corredores para esta carrera.")
    
    if 'final_bets' in st.session_state and st.session_state.final_bets:
        st.subheader("Apuestas Recomendadas para Mañana")
        
        with st.form("bets_form"):
            for i, bet in enumerate(st.session_state.final_bets):
                st.markdown(f"**{bet['horse_name']}** - {bet['course']} {bet['time']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Tipo de Apuesta", bet['stake_type'])
                col2.metric("Cuota IA", f"{bet['ia_odds']:.2f}")
                st.session_state.final_bets[i]['placed_odds'] = col3.number_input("Cuota Apostada", min_value=1.0, value=bet['ia_odds'], step=0.1, key=f"odds_{i}")
                st.divider()
            
            if st.form_submit_button("Guardar Apuestas Realizadas"):
                new_bets = []
                for bet in st.session_state.final_bets:
                    bet_id = f"{int(time.time())}-{bet['horse_name']}"
                    new_bet = {
                        'bet_id': bet_id, 'horse_name': bet['horse_name'],
                        'course': bet['course'], 'time': bet['time'],
                        'stake_type': bet['stake_type'], 'stake_points': bet['stake_points'],
                        'ia_odds': bet['ia_odds'], 'placed_odds': bet['placed_odds'],
                        'status': 'Pendiente', 'pnl': 0.0
                    }
                    new_bets.append(new_bet)
                
                new_bets_df = pd.DataFrame(new_bets)
                st.session_state.historical_bets = pd.concat([st.session_state.historical_bets, new_bets_df], ignore_index=True)
                st.success(f"{len(new_bets)} apuestas guardadas con éxito.")
                del st.session_state.final_bets
                st.rerun()

# --- PESTAÑA 2: GESTIONAR APUESTAS PENDIENTES ---
with tab2:
    st.header("Gestionar Resultados de Apuestas")
    pending_bets = st.session_state.historical_bets[st.session_state.historical_bets['status'] == 'Pendiente']
    
    if pending_bets.empty:
        st.info("No hay apuestas pendientes de resultado.")
    else:
        for index, bet in pending_bets.iterrows():
            st.markdown(f"**{bet['horse_name']}** ({bet['course']} {bet['time']}) - **Cuota:** {bet['placed_odds']:.2f}")
            st.selectbox(
                "Resultado:",
                ("Pendiente", "Ganada", "Perdida", "Colocada", "No Corrió"),
                key=f"status_{bet['bet_id']}",
                on_change=update_bet_status,
                args=(bet['bet_id'],)
            )
            st.divider()

# --- PESTAÑA 3: HISTORIAL DE RESULTADOS ---
with tab3:
    st.header("Historial de Todas las Apuestas")
    settled_bets = st.session_state.historical_bets[st.session_state.historical_bets['status'] != 'Pendiente']
    
    if settled_bets.empty:
        st.info("No hay apuestas con resultados definidos.")
    else:
        color_map = {"Ganada": "green", "Perdida": "red", "Colocada": "blue", "No Corrió": "gray"}
        for index, bet in settled_bets.iterrows():
            color = color_map.get(bet['status'], "black")
            st.markdown(f"**<font color='{color}'>{bet['status']}:</font>** {bet['horse_name']} en {bet['course']} - **P/L: {bet['pnl']:.2f} PT**", unsafe_allow_html=True)

# --- PESTAÑA 4: RENDIMIENTO GENERAL ---
with tab4:
    st.header("Rendimiento Histórico")
    
    settled_bets_perf = st.session_state.historical_bets[st.session_state.historical_bets['status'] != 'Pendiente']
    
    if settled_bets_perf.empty:
        st.info("No hay datos suficientes para calcular el rendimiento.")
    else:
        total_pnl = settled_bets_perf['pnl'].sum()
        total_staked = settled_bets_perf['stake_points'].sum()
        roi = (total_pnl / total_staked) * 100 if total_staked > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Beneficio/Pérdida Total (PT)", f"{total_pnl:.2f}")
        col2.metric("ROI (Retorno de la Inversión)", f"{roi:.2f}%")
        
        st.subheader("Desglose de Apuestas")
        st.dataframe(settled_bets_perf[['horse_name', 'course', 'placed_odds', 'status', 'pnl']])
