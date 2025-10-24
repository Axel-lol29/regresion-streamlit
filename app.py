import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regresión lineal simple", page_icon="📈", layout="centered")

st.title("📈 Regresión lineal simple")
st.write(
    "Sube un **CSV** con columnas numéricas (X e y). Elige las columnas, entrena el modelo, "
    "mira la recta de regresión y haz una predicción rápida."
)

# ---------- Carga de datos ----------
file = st.file_uploader("Sube tu CSV", type=["csv"])

@st.cache_data
def load_sample():
    rng = np.random.default_rng(42)
    x = np.linspace(1, 10, 40)
    y = 10 + 8 * x + rng.normal(0, 6, size=len(x))
    return pd.DataFrame({"horas_estudio": x, "calificacion": y})

if file:
    df = pd.read_csv(file)
    st.success("Datos cargados correctamente.")
else:
    st.info("No subiste archivo. Usando **datos de ejemplo** (horas_estudio → calificacion).")
    df = load_sample()

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# ---------- Selección de columnas numéricas ----------
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) < 2:
    st.error("Se requieren **al menos dos columnas numéricas** (X e y).")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    x_col = st.selectbox("Variable independiente (X)", num_cols, index=0)
with c2:
    y_col = st.selectbox("Variable dependiente (y)", [c for c in num_cols if c != x_col], index=0)

# ---------- Entrenamiento ----------
data = df[[x_col, y_col]].dropna()  # quita NAs alineados
X = data[[x_col]].values
y = data[y_col].values

model = LinearRegression().fit(X, y)
y_hat = model.predict(X)

# ---------- Métricas y ecuación ----------
st.subheader("Modelo y métricas")
st.write(f"**Ecuación:**  y = {model.intercept_:.4f} + {model.coef_[0]:.4f} · x")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{r2_score(y, y_hat):.4f}")
m2.metric("MAE", f"{mean_absolute_error(y, y_hat):.4f}")
m3.metric("MSE", f"{mean_squared_error(y, y_hat):.4f}")
m4.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_hat)):.4f}")

# ---------- Gráfica (se refresca) ----------
st.subheader("Dispersión y recta de regresión")
fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.7)
x_sorted = np.sort(X.reshape(-1))
ax.plot(x_sorted, model.predict(x_sorted.reshape(-1, 1)), linewidth=2)
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title("Ajuste de regresión")
st.pyplot(fig, clear_figure=True)

# ---------- Predicción rápida (sin segundo CSV) ----------
st.subheader("Predicción rápida")
val = st.number_input(f"Nuevo valor para `{x_col}`", value=float(np.median(X)))
pred = model.predict(np.array([[val]]))[0]
st.write(f"**Predicción de `{y_col}`:** {pred:.4f}")

st.caption("Tip: si luego necesitas 'aceptar nuevos datos' en lote, agrego un uploader opcional para un CSV solo con la columna X.")
