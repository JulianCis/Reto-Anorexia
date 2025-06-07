import streamlit as st
import joblib
import os

st.set_page_config(page_title="Clasificador de Tweets", layout="centered")
st.title("🧠 Clasificador de Tweets: Anorexia vs Control")
st.write("Clasifica un tweet como relacionado a **anorexia** o **control saludable**.")

@st.cache_resource
def load_pipeline():
    path = os.path.expanduser("~/Desktop/ia/pipeline.pkl")  # ajusta si lo guardaste en otro lugar
    if not os.path.exists(path):
        st.error("❌ No se encontró `pipeline.pkl`. Colócalo en el Escritorio o ajusta la ruta.")
        st.stop()
    return joblib.load(path)

pipeline = load_pipeline()

tweet = st.text_area("✏️ Escribe un tweet:", height=120, placeholder="Ejemplo: Me comí solo una manzana...")

if tweet.strip():
    pred = pipeline.predict([tweet])[0]
    prob = pipeline.predict_proba([tweet]).max()

    st.markdown("### Resultado:")
    st.success(f"📝 Clasificado como **{pred.upper()}** con una confianza de **{prob:.2%}**.")
