import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regresión lineal simple", page_icon="📈", layout="centered")

st.title("📈 Regresión lineal simple (Streamlit)")
st.write(
    "Carga un **CSV** con columnas numéricas, elige la variable independiente (X) y la dependiente (y), "
    "entrena el modelo y genera predicciones con datos nuevos."
)

# ============== Cargar datos ==============
uploaded = st.file_uploader("Sube tu archivo CSV con los datos de entrenamiento", type=["csv"])

@st.cache_data
def load_sample():
    # Datos de ejemplo: Horas de estudio vs Calificación
    rng = np.random.default_rng(42)
    x = np.linspace(1, 10, 40)
    y = 10 + 8*x + rng.normal(0, 6, size=len(x))
    df = pd.DataFrame({"horas_estudio": x, "calificacion": y})
    return df

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Datos cargados correctamente.")
else:
    st.info("No subiste archivo. Usando **datos de ejemplo** (horas_estudio → calificacion).")
    df = load_sample()

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Filtrar columnas numéricas
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Se requieren **al menos dos columnas numéricas** para hacer regresión (X e y).")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    x_col = st.selectbox("Variable independiente (X)", numeric_cols, index=0)
with col2:
    y_col = st.selectbox("Variable dependiente (y)", [c for c in numeric_cols if c != x_col], index=0)

# ============== Entrenar modelo ==============
X = df[[x_col]].dropna().values
y = df[y_col].dropna().values
if len(X) != len(y):
    st.warning("Hay valores faltantes desalineados entre X e y. Se eliminarán filas con NA.")
    cleaned = df[[x_col, y_col]].dropna()
    X = cleaned[[x_col]].values
    y = cleaned[y_col].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.subheader("Parámetros del modelo")
st.write(f"**Intercepto (β₀):** {model.intercept_:.4f}")
st.write(f"**Pendiente (β₁) para `{x_col}`:** {model.coef_[0]:.4f}")

st.subheader("Métricas (sobre los datos usados para entrenar)")
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
met_cols = st.columns(4)
met_cols[0].metric("R²", f"{r2:.4f}")
met_cols[1].metric("MAE", f"{mae:.4f}")
met_cols[2].metric("MSE", f"{mse:.4f}")
met_cols[3].metric("RMSE", f"{rmse:.4f}")

# ============== Gráfico ==============
st.subheader("Dispersión y recta de regresión")
fig = plt.figure()
plt.scatter(X, y, alpha=0.7)
x_sorted = np.sort(X.reshape(-1))
plt.plot(x_sorted, model.predict(x_sorted.reshape(-1, 1)), linewidth=2)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Ajuste de regresión lineal")
st.pyplot(fig)

# ============== Predicción puntual ==============
st.subheader("Predicción con un valor nuevo de X")
new_x = st.number_input(f"Ingresar un valor para `{x_col}`", value=float(np.median(X)))
pred_single = model.predict(np.array([[new_x]]))[0]
st.write(f"**Predicción de `{y_col}`:** {pred_single:.4f}")

# ============== Predicciones por archivo ==============
st.subheader("Subir archivo con nuevos datos de X para predecir y descargar resultados")
st.caption(f"El archivo debe tener una columna llamada **{x_col}** (CSV).")
new_file = st.file_uploader("Sube CSV con nuevos valores de X", type=["csv"], key="newcsv")

if new_file is not None:
    new_df = pd.read_csv(new_file)
    if x_col not in new_df.columns:
        st.error(f"El archivo debe contener la columna `{x_col}`.")
    else:
        out = new_df.copy()
        out[f"pred_{y_col}"] = model.predict(out[[x_col]].values)
        st.write("Vista previa de predicciones:")
        st.dataframe(out.head())
        buffer = io.StringIO()
        out.to_csv(buffer, index=False)
        st.download_button("⬇️ Descargar predicciones CSV", data=buffer.getvalue(),
                           file_name="predicciones.csv", mime="text/csv")

st.divider()
st.caption("Hecho con ❤️ en Streamlit.")
