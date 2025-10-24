import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import altair as alt

# -------------------- Configuración básica --------------------
st.set_page_config(page_title="Regresión lineal simple", page_icon="📈", layout="centered")

st.title("📈 Regresión lineal simple (By Axel Mireles #739047)")
st.caption("Sube un CSV con dos columnas numéricas (X e y), entrena el modelo, visualiza la recta e intenta una predicción rápida.")

# -------------------- Carga de datos --------------------
file = st.file_uploader("Sube tu CSV (ej. columnas: x, y)", type=["csv"])

@st.cache_data
def load_sample():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 50, 60)
    y = 12 + 3.2 * x + rng.normal(0, 18, size=len(x))
    return pd.DataFrame({"x": x, "y": y})

if file:
    df = pd.read_csv(file)
    st.success("Datos cargados correctamente.")
else:
    st.info("No subiste archivo. Usando **datos de ejemplo** (x → y). Puedes descargar un CSV de prueba desde la tarea.")
    df = load_sample()

st.subheader("Vista previa")
st.dataframe(df.head())

# -------------------- Selección de columnas --------------------
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) < 2:
    st.error("Se requieren al menos **dos columnas numéricas** (X e y).")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    x_col = st.selectbox("Variable independiente (X)", num_cols, index=0, key="xcol")
with c2:
    y_col = st.selectbox("Variable dependiente (y)", [c for c in num_cols if c != x_col], index=0, key="ycol")

# -------------------- Limpieza & modelo --------------------
data = df[[x_col, y_col]].dropna().rename(columns={x_col: "X", y_col: "Y"})
if data.empty or data["X"].nunique() < 2:
    st.error("No hay suficientes datos válidos para entrenar (revisa NAs y que X tenga variación).")
    st.stop()

X = data[["X"]].values
Y = data["Y"].values

model = LinearRegression().fit(X, Y)
Y_hat = model.predict(X)

# -------------------- Métricas y ecuación --------------------
st.subheader("Modelo y métricas")
st.write(f"**Ecuación:**  y = {model.intercept_:.4f} + {model.coef_[0]:.4f} · x")

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{r2_score(Y, Y_hat):.4f}")
m2.metric("MAE", f"{mean_absolute_error(Y, Y_hat):.4f}")
m3.metric("MSE", f"{mean_squared_error(Y, Y_hat):.4f}")
m4.metric("RMSE", f"{np.sqrt(mean_squared_error(Y, Y_hat)):.4f}")

# -------------------- Gráfica interactiva (Altair) --------------------
st.subheader("Dispersión y recta de regresión (interactiva)")

# Puntos
points = alt.Chart(data).mark_circle(size=70, opacity=0.8).encode(
    x=alt.X("X:Q", title=x_col),
    y=alt.Y("Y:Q", title=y_col),
    tooltip=[alt.Tooltip("X:Q", title=x_col), alt.Tooltip("Y:Q", title=y_col)]
)

# Línea (evaluar el modelo en rango continuo de X)
x_min, x_max = float(data["X"].min()), float(data["X"].max())
x_line = np.linspace(x_min, x_max, 200).reshape(-1, 1)
line_df = pd.DataFrame({"X": x_line.ravel(), "Y_pred": model.predict(x_line)})

line = alt.Chart(line_df).mark_line(size=3).encode(
    x=alt.X("X:Q", title=x_col),
    y=alt.Y("Y_pred:Q", title=y_col),
    tooltip=[alt.Tooltip("X:Q", title=x_col), alt.Tooltip("Y_pred:Q", title=f"pred_{y_col}")]
)

st.altair_chart((points + line).properties(height=420).interactive(), use_container_width=True)

# -------------------- Predicción rápida --------------------
st.subheader("Predicción rápida")
default_val = float(np.median(data["X"]))
new_x = st.number_input(f"Nuevo valor para `{x_col}`", value=default_val)
pred = model.predict(np.array([[new_x]]))[0]
st.write(f"**Predicción de `{y_col}`:** {pred:.4f}")

st.caption("Tip: cambia las columnas o sube otro CSV y verás la gráfica y las métricas actualizarse al instante.")
