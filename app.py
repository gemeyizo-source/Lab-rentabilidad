
# ================================================================
# MODELO DINAMICO DE RENTABILIDAD MEDIANTE SUCESIONES Y SERIES
# Aplicacion interactiva con Streamlit
# Compatible con Visual Studio Code
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import gamma
from io import BytesIO

# ================================================================
# 1. CONFIGURACION GENERAL
# ================================================================

st.set_page_config(
    page_title="Modelo matematico de rentabilidad",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Modelo matematico dinamico de rentabilidad")
st.markdown("""
**Laboratorio de sucesiones, series, capitalizacion compuesta,
riesgo financiero y convergencia.**

Modifique los parametros y observe como cambia el comportamiento
matematico y financiero de las estrategias.
""")

st.divider()

# ================================================================
# 2. FUNCIONES MATEMATICAS
# ================================================================

def generar_rendimientos(
    N, r0, p, estrategia, sigma=0.0,
    prob_perdida=0.0, magnitud_perdida=0.0,
    semilla=42
):
    """
    Genera rendimientos positivos, alternantes o estocasticos.
    Los rendimientos se generan para periodos k=1,...,N.
    """

    k = np.arange(1, N + 1)
    base = r0 / (k + 1) ** p

    if estrategia == "Positiva decreciente":
        r = base

    elif estrategia == "Alternante":
        r = ((-1.0) ** k) * base

    else:
        rng = np.random.default_rng(semilla)

        ruido = rng.normal(0, sigma, N)

        eventos = rng.random(N) < prob_perdida

        perdidas = eventos * magnitud_perdida

        r = base + ruido - perdidas

    return r


def calcular_capital(
    W0, rendimientos, costo, impuesto, inflacion
):
    """
    Capital nominal y real.

    Impuesto: se aplica sobre rendimientos positivos.
    Costo: proporcion del capital cobrada cada periodo.
    Inflacion: tasa por periodo.
    """

    N = len(rendimientos)

    capital = np.zeros(N + 1)
    capital_real = np.zeros(N + 1)

    factor_log = np.zeros(N)

    capital[0] = W0
    capital_real[0] = W0

    costos_periodo = np.zeros(N)
    impuestos_periodo = np.zeros(N)

    for i, r in enumerate(rendimientos):

        # Validar rendimiento
        if r <= -1:
            raise ValueError(
                "Existe un rendimiento menor o igual a -100 %."
            )

        if inflacion <= -1:
            raise ValueError("Inflacion no valida.")

        if not (0 <= costo < 1):
            raise ValueError("El costo debe estar entre 0 y 1.")

        if not (0 <= impuesto <= 1):
            raise ValueError("El impuesto debe estar entre 0 y 1.")

        # Impuesto sobre ganancias positivas
        impuesto_tasa = impuesto * max(r, 0)

        # Factor despues de impuestos
        factor_despues_impuesto = 1 + r - impuesto_tasa

        # Factor despues de costos
        factor_neto = factor_despues_impuesto * (1 - costo)

        if factor_neto <= 0:
            raise ValueError("Factor neto no positivo.")

        # Capital nominal
        capital[i + 1] = capital[i] * factor_neto

        # Capital real
        capital_real[i + 1] = (
            capital_real[i] * factor_neto / (1 + inflacion)
        )

        # Crecimiento logaritmico neto
        factor_log[i] = np.log(factor_neto)

        costos_periodo[i] = capital[i] * (
            factor_despues_impuesto * costo
        )

        impuestos_periodo[i] = capital[i] * (
            impuesto_tasa * (1 - costo)
        )

    return (
        capital, capital_real, factor_log,
        costos_periodo, impuestos_periodo
    )


def max_drawdown(capital):
    """Calcula la mayor caida desde un maximo anterior."""

    maximos = np.maximum.accumulate(capital)

    drawdown = capital / maximos - 1

    return drawdown


def clasificar_serie(p, estrategia):
    """Clasificacion teorica para las familias de rendimientos."""

    if p <= 0:
        return "No converge a cero."

    if estrategia == "Positiva decreciente":

        if p > 1:
            return "Convergente."
        else:
            return "Divergente."

    if estrategia == "Alternante":

        if p > 1:
            return "Convergencia absoluta."
        else:
            return "Convergencia condicional."

    return "No existe garantia teorica: proceso estocastico."


def clasificar_capital(p, estrategia):

    if estrategia == "Positiva decreciente":

        if p > 1:
            return "Capital con limite positivo y finito."

        if p > 0:
            return "Capital creciente sin limite."

        return "No se garantiza convergencia."

    if estrategia == "Alternante":

        if p > 0.5:
            return "Capital con limite positivo y finito."

        if p > 0:
            return "Capital tiende a cero."

        return "No se garantiza convergencia."

    return "Depende de la distribucion y parametros aleatorios."


def crear_excel(df, resumen):
    """Genera archivo Excel con dos hojas."""

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

        df.to_excel(
            writer,
            sheet_name="Resultados",
            index=False
        )

        resumen.to_excel(
            writer,
            sheet_name="Indicadores",
            index=False
        )

    buffer.seek(0)

    return buffer


# ================================================================
# 3. PANEL DE PARAMETROS
# ================================================================

st.sidebar.header("⚙️ Configuracion del modelo")

st.sidebar.subheader("Capital y rendimientos")

W0 = st.sidebar.number_input(
    "Capital inicial ($)",
    min_value=100.0,
    max_value=1000000000.0,
    value=1000000.0,
    step=10000.0
)

r0 = st.sidebar.slider(
    "Rendimiento base (%)",
    min_value=0.1,
    max_value=50.0,
    value=10.0,
    step=0.5
) / 100

p = st.sidebar.slider(
    "Exponente de decrecimiento (p)",
    min_value=0.05,
    max_value=3.0,
    value=1.0,
    step=0.05
)

N = st.sidebar.slider(
    "Numero de periodos",
    min_value=10,
    max_value=5000,
    value=500,
    step=10
)

estrategia = st.sidebar.selectbox(
    "Tipo de estrategia",
    [
        "Positiva decreciente",
        "Alternante",
        "Estocastica"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Riesgo e incertidumbre")

sigma = st.sidebar.slider(
    "Volatilidad por periodo (%)",
    0.0, 30.0, 2.0, 0.5
) / 100

prob_perdida = st.sidebar.slider(
    "Probabilidad de perdida (%)",
    0.0, 100.0, 10.0, 1.0
) / 100

magnitud_perdida = st.sidebar.slider(
    "Magnitud adicional de perdida (%)",
    0.0, 90.0, 5.0, 1.0
) / 100

semilla = st.sidebar.number_input(
    "Semilla aleatoria",
    min_value=1,
    max_value=999999,
    value=42
)

st.sidebar.divider()
st.sidebar.subheader("Costos e inflacion")

costo = st.sidebar.slider(
    "Comision por periodo (%)",
    0.0, 10.0, 0.1, 0.05
) / 100

impuesto = st.sidebar.slider(
    "Impuesto sobre ganancias (%)",
    0.0, 50.0, 10.0, 1.0
) / 100

inflacion = st.sidebar.slider(
    "Inflacion por periodo (%)",
    0.0, 20.0, 0.3, 0.1
) / 100

st.sidebar.divider()

st.sidebar.caption(
    "Los porcentajes de inflacion, costos e impuestos "
    "se interpretan por periodo, no necesariamente por año."
)

# ================================================================
# 4. GENERAR DATOS
# ================================================================

try:

    rendimientos = generar_rendimientos(
        N=N,
        r0=r0,
        p=p,
        estrategia=estrategia,
        sigma=sigma,
        prob_perdida=prob_perdida,
        magnitud_perdida=magnitud_perdida,
        semilla=int(semilla)
    )

    (
        capital,
        capital_real,
        log_rendimientos,
        costos_periodo,
        impuestos_periodo
    ) = calcular_capital(
        W0,
        rendimientos,
        costo,
        impuesto,
        inflacion
    )

except ValueError as error:

    st.error(str(error))
    st.stop()

# ================================================================
# 5. CONSTRUCCION DE INDICADORES
# ================================================================

periodos = np.arange(1, N + 1)

suma_rendimientos = np.cumsum(rendimientos)

suma_log = np.cumsum(log_rendimientos)

drawdown = max_drawdown(capital)

volatilidad = np.std(rendimientos, ddof=1)

rendimiento_acumulado = capital[-1] / W0 - 1

rendimiento_real = capital_real[-1] / W0 - 1

maxima_caida = np.min(drawdown)

ganancias_periodos = np.sum(rendimientos > 0)

perdidas_periodos = np.sum(rendimientos < 0)

# ================================================================
# 6. ENCABEZADO DE RESULTADOS
# ================================================================

st.subheader("📊 Indicadores principales")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Capital nominal final",
    f"${capital[-1]:,.2f}",
    f"{rendimiento_acumulado*100:.2f}%"
)

col2.metric(
    "Capital real final",
    f"${capital_real[-1]:,.2f}",
    f"{rendimiento_real*100:.2f}%"
)

col3.metric(
    "Volatilidad por periodo",
    f"{volatilidad*100:.2f}%"
)

col4.metric(
    "Maximo drawdown",
    f"{maxima_caida*100:.2f}%"
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Rendimiento promedio",
    f"{np.mean(rendimientos)*100:.4f}%"
)

col6.metric(
    "Ganancias",
    int(ganancias_periodos)
)

col7.metric(
    "Periodos negativos",
    int(perdidas_periodos)
)

col8.metric(
    "Crecimiento logaritmico",
    f"{suma_log[-1]:.4f}"
)

st.divider()

# ================================================================
# 7. DIAGNOSTICO MATEMATICO
# ================================================================

st.subheader("🧮 Diagnostico matematico")

a, b, c = st.columns(3)

a.info(
    "**Convergencia de la sucesion**\n\n"
    + (
        "El rendimiento tiende a cero."
        if p > 0
        else "El rendimiento no tiende a cero."
    )
)

b.info(
    "**Convergencia de la serie**\n\n"
    + clasificar_serie(p, estrategia)
)

c.info(
    "**Comportamiento teorico del capital**\n\n"
    + clasificar_capital(p, estrategia)
)

st.caption(
    "La clasificacion teorica corresponde a las familias "
    "deterministas sin costos ni impuestos. La simulacion "
    "observada incorpora los parametros seleccionados."
)

# ================================================================
# 8. TABLAS DE NAVEGACION
# ================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Rendimientos",
    "💰 Capital",
    "📐 Convergencia",
    "🎲 Monte Carlo",
    "🔬 Sensibilidad",
    "📋 Datos y discusion"
])

# ================================================================
# TAB 1: RENDIMIENTOS
# ================================================================

with tab1:

    st.subheader("Comportamiento de la sucesion")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=periodos,
        y=rendimientos * 100,
        mode="lines",
        name="Rendimiento",
        line=dict(color="#1677C8", width=2)
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig.update_layout(
        title="Rendimiento por periodo",
        xaxis_title="Periodo",
        yaxis_title="Rendimiento (%)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=periodos,
        y=suma_rendimientos,
        mode="lines",
        name="Suma acumulada",
        line=dict(color="#16A078", width=2)
    ))

    fig2.update_layout(
        title="Suma parcial de rendimientos",
        xaxis_title="Periodo",
        yaxis_title="Suma acumulada",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Interpretacion:**

    - La primera grafica permite observar si los rendimientos
      disminuyen, alternan o presentan volatilidad.
    - La segunda permite observar si la suma acumulada parece
      estabilizarse o continuar creciendo.
    - Un horizonte finito no demuestra convergencia matematica.
    """)

# ================================================================
# TAB 2: CAPITAL
# ================================================================

with tab2:

    st.subheader("Evolucion del capital")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(N + 1),
        y=capital,
        mode="lines",
        name="Capital nominal",
        line=dict(color="#1677C8", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(N + 1),
        y=capital_real,
        mode="lines",
        name="Capital real",
        line=dict(color="#16A078", width=3)
    ))

    fig.update_layout(
        title="Capital nominal frente al capital real",
        xaxis_title="Periodo",
        yaxis_title="Capital ($)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_log = go.Figure()

    fig_log.add_trace(go.Scatter(
        x=periodos,
        y=suma_log,
        mode="lines",
        name="Suma logaritmica"
    ))

    fig_log.update_layout(
        title="Crecimiento logaritmico acumulado",
        xaxis_title="Periodo",
        yaxis_title="Logaritmo acumulado",
        template="plotly_white"
    )

    st.plotly_chart(fig_log, use_container_width=True)

    fig_dd = go.Figure()

    fig_dd.add_trace(go.Scatter(
        x=np.arange(N + 1),
        y=max_drawdown(capital) * 100,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#D9534F")
    ))

    fig_dd.update_layout(
        title="Riesgo de caida del capital",
        xaxis_title="Periodo",
        yaxis_title="Caida desde maximo (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig_dd, use_container_width=True)

# ================================================================
# TAB 3: CONVERGENCIA
# ================================================================

with tab3:

    st.subheader("Analisis de convergencia")

    k = np.arange(1, N + 1)

    col_a, col_b = st.columns(2)

    with col_a:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=k,
            y=suma_rendimientos,
            name="Suma de rendimientos"
        ))

        fig.update_layout(
            title="Convergencia de la serie",
            xaxis_title="Periodo",
            yaxis_title="Suma parcial",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_b:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=k,
            y=suma_log,
            name="Serie de log-rendimientos",
            line=dict(color="#16A078")
        ))

        fig.update_layout(
            title="Convergencia de log-rendimientos",
            xaxis_title="Periodo",
            yaxis_title="Suma logaritmica",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Umbrales teoricos")

    st.markdown("""
    Para rendimientos positivos de tipo potencia:

    - Si `p > 1`, converge la suma de rendimientos.
    - Si `0 < p <= 1`, la suma diverge.
    - El capital positivo converge a un limite finito cuando `p > 1`.

    Para rendimientos alternantes:

    - Si `p > 0`, converge la serie alternante.
    - Si `p > 1`, converge absolutamente.
    - Si `0 < p <= 1`, converge condicionalmente.
    - Bajo las condiciones del modelo, el capital tiene limite
      positivo cuando `p > 0.5`.
    """)

    st.warning(
        "No confundir convergencia numerica aparente con "
        "demostracion analitica de convergencia."
    )

# ================================================================
# TAB 4: SIMULACION MONTE CARLO
# ================================================================

with tab4:

    st.subheader("Simulacion Monte Carlo")

    st.markdown("""
    Se generan multiples trayectorias de inversion para estudiar
    la incertidumbre del capital futuro.
    """)

    numero_simulaciones = st.slider(
        "Numero de simulaciones",
        100,
        5000,
        1000,
        100
    )

    rng = np.random.default_rng(int(semilla))

    trayectorias = np.zeros(
        (numero_simulaciones, N + 1)
    )

    trayectorias[:, 0] = W0

    for j in range(numero_simulaciones):

        ruido = rng.normal(0, sigma, N)

        eventos = (
            rng.random(N) < prob_perdida
        )

        r_mc = (
            r0 / (k + 1) ** p
            + ruido
            - eventos * magnitud_perdida
        )

        r_mc = np.maximum(r_mc, -0.999)

        factor_mc = (
            1 + r_mc - impuesto * np.maximum(r_mc, 0)
        ) * (1 - costo)

        trayectorias[j, 1:] = (
            W0 * np.cumprod(factor_mc)
        )

    percentiles = np.percentile(
        trayectorias,
        [5, 25, 50, 75, 95],
        axis=0
    )

    fig_mc = go.Figure()

    eje = np.arange(N + 1)

    fig_mc.add_trace(go.Scatter(
        x=eje,
        y=percentiles[4],
        name="Percentil 95",
        line=dict(color="lightblue")
    ))

    fig_mc.add_trace(go.Scatter(
        x=eje,
        y=percentiles[0],
        name="Percentil 5",
        fill="tonexty",
        fillcolor="rgba(100,150,200,0.15)",
        line=dict(color="lightblue")
    ))

    fig_mc.add_trace(go.Scatter(
        x=eje,
        y=percentiles[2],
        name="Mediana",
        line=dict(color="#D9534F", width=3)
    ))

    fig_mc.update_layout(
        title="Distribucion del capital simulado",
        xaxis_title="Periodo",
        yaxis_title="Capital nominal ($)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    capital_final_mc = trayectorias[:, -1]

    mc1, mc2, mc3 = st.columns(3)

    mc1.metric(
        "Capital mediano",
        f"${np.median(capital_final_mc):,.2f}"
    )

    mc2.metric(
        "Percentil 5",
        f"${np.percentile(capital_final_mc, 5):,.2f}"
    )

    mc3.metric(
        "Probabilidad de perdida",
        f"{np.mean(capital_final_mc < W0)*100:.2f}%"
    )

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=capital_final_mc,
        nbinsx=40,
        marker_color="#1677C8"
    ))

    fig_hist.update_layout(
        title="Distribucion del capital final",
        xaxis_title="Capital final ($)",
        yaxis_title="Frecuencia",
        template="plotly_white"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "Monte Carlo utiliza un proceso de rendimientos "
        "estocasticos especificado por los parametros. "
        "No representa una prediccion garantizada."
    )

# ================================================================
# TAB 5: ANALISIS DE SENSIBILIDAD
# ================================================================

with tab5:

    st.subheader("Analisis de sensibilidad del parametro p")

    valores_p = np.linspace(0.1, 3.0, 35)

    capitales_sensibilidad = []

    for p_test in valores_p:

        r_test = r0 / (k + 1) ** p_test

        factor_test = (
            1 + r_test - impuesto * np.maximum(r_test, 0)
        ) * (1 - costo)

        capital_test = W0 * np.prod(factor_test)

        capitales_sensibilidad.append(capital_test)

    fig_sens = go.Figure()

    fig_sens.add_trace(go.Scatter(
        x=valores_p,
        y=capitales_sensibilidad,
        mode="lines+markers",
        name="Capital final"
    ))

    fig_sens.add_vline(
        x=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Umbral p = 1"
    )

    fig_sens.update_layout(
        title="Sensibilidad del capital frente a p",
        xaxis_title="Parametro p",
        yaxis_title="Capital final ($)",
        template="plotly_white"
    )

    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("### Sensibilidad del capital frente a costos")

    costos_test = np.linspace(0, 0.05, 30)

    capitales_costos = []

    for c_test in costos_test:

        r_test = rendimientos

        factor_test = (
            1 + r_test - impuesto * np.maximum(r_test, 0)
        ) * (1 - c_test)

        capital_test = W0 * np.prod(factor_test)

        capitales_costos.append(capital_test)

    fig_costos = go.Figure()

    fig_costos.add_trace(go.Scatter(
        x=costos_test * 100,
        y=capitales_costos,
        mode="lines+markers",
        line=dict(color="#D9534F")
    ))

    fig_costos.update_layout(
        title="Efecto de los costos sobre el capital",
        xaxis_title="Costo por periodo (%)",
        yaxis_title="Capital final ($)",
        template="plotly_white"
    )

    st.plotly_chart(fig_costos, use_container_width=True)

    st.markdown("### Sensibilidad a la inflacion")

    inflaciones_test = np.linspace(0, 0.05, 30)

    capitales_reales = []

    for pi_test in inflaciones_test:

        capital_real_test = capital[-1] / (1 + pi_test) ** N

        capitales_reales.append(capital_real_test)

    fig_inf = go.Figure()

    fig_inf.add_trace(go.Scatter(
        x=inflaciones_test * 100,
        y=capitales_reales,
        mode="lines+markers",
        line=dict(color="#16A078")
    ))

    fig_inf.update_layout(
        title="Efecto de la inflacion sobre el capital real",
        xaxis_title="Inflacion por periodo (%)",
        yaxis_title="Capital real final ($)",
        template="plotly_white"
    )

    st.plotly_chart(fig_inf, use_container_width=True)

# ================================================================
# TAB 6: DATOS, INDICADORES Y DISCUSION
# ================================================================

with tab6:

    st.subheader("Tabla de resultados")

    df = pd.DataFrame({

        "Periodo": periodos,

        "Rendimiento": rendimientos,

        "Rendimiento (%)": rendimientos * 100,

        "Suma acumulada": suma_rendimientos,

        "Log-rendimiento neto": log_rendimientos,

        "Log acumulado": suma_log,

        "Capital nominal": capital[1:],

        "Capital real": capital_real[1:],

        "Drawdown": drawdown[1:],

        "Costos ($)": costos_periodo,

        "Impuestos ($)": impuestos_periodo

    })

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

    # Indicadores para exportacion
    resumen = pd.DataFrame({

        "Indicador": [
            "Capital inicial",
            "Capital nominal final",
            "Capital real final",
            "Rendimiento acumulado",
            "Rendimiento real acumulado",
            "Rendimiento promedio",
            "Volatilidad",
            "Maximo drawdown",
            "Suma de rendimientos",
            "Crecimiento logaritmico",
            "Periodos positivos",
            "Periodos negativos"
        ],

        "Valor": [
            W0,
            capital[-1],
            capital_real[-1],
            rendimiento_acumulado,
            rendimiento_real,
            np.mean(rendimientos),
            volatilidad,
            maxima_caida,
            suma_rendimientos[-1],
            suma_log[-1],
            ganancias_periodos,
            perdidas_periodos
        ]
    })

    st.subheader("Indicadores consolidados")

    st.dataframe(
        resumen,
        use_container_width=True
    )

    # Discusion automatica
    st.subheader("🧠 Generador de discusion de resultados")

    st.markdown("### Interpretacion matematica")

    if estrategia == "Positiva decreciente":

        if p > 1:

            st.success("""
            El parametro p es superior a uno. La serie de
            rendimientos positivos converge y el producto
            de capitalizacion tiene un limite finito positivo,
            bajo los supuestos del modelo.
            """)

        else:

            st.warning("""
            El parametro p es menor o igual a uno. Aunque
            los rendimientos individuales tienden a cero,
            la suma de rendimientos diverge y el capital
            nominal crece sin limite en el modelo sin costos.
            """)

    elif estrategia == "Alternante":

        if p > 1:

            st.success("""
            La serie alternante converge absolutamente.
            La capitalizacion presenta un limite positivo
            y finito bajo las condiciones establecidas.
            """)

        elif p > 0.5:

            st.info("""
            La serie alternante converge condicionalmente.
            El capital tambien converge a un valor positivo
            y finito, aunque no debe confundirse la suma
            de rendimientos con la capitalizacion.
            """)

        elif p > 0:

            st.warning("""
            La serie alternante converge condicionalmente,
            pero el capital tiende a cero por el efecto
            acumulativo de los terminos cuadraticos
            en los log-rendimientos.
            """)

    else:

        st.info("""
        La estrategia estocastica no posee un umbral de
        convergencia determinado exclusivamente por p.
        Su comportamiento depende de la distribucion,
        volatilidad, probabilidad de perdida y magnitud
        de los choques.
        """)

    st.markdown("### Interpretacion financiera")

    if capital[-1] > W0:

        st.write(
            "El capital nominal final supera el capital inicial. "
            "Existe crecimiento nominal en el horizonte analizado."
        )

    else:

        st.write(
            "El capital nominal final es inferior al capital inicial. "
            "La estrategia presenta perdida nominal acumulada."
        )

    if capital_real[-1] < W0:

        st.write(
            "Al descontar la inflacion, se observa perdida de "
            "poder adquisitivo respecto al capital inicial."
        )

    else:

        st.write(
            "El capital real supera el capital inicial y conserva "
            "un crecimiento positivo en poder adquisitivo."
        )

    st.markdown("### Preguntas para la discusion academica")

    preguntas = [
        "¿Como cambia el capital cuando aumenta el parametro p?",

        "¿Por que un rendimiento que tiende a cero no garantiza "
        "que la suma de rendimientos converja?",

        "¿Que diferencia existe entre convergencia absoluta "
        "y convergencia condicional?",

        "¿Como afectan los costos periodicos al capital final?",

        "¿En que medida la inflacion reduce el rendimiento real?",

        "¿Por que una serie alternante convergente puede "
        "producir un capital que tiende a cero?",

        "¿Como cambia el riesgo cuando aumenta la volatilidad?",

        "¿Que ocurre cuando aumenta la probabilidad de perdidas?",

        "¿Que estrategia presenta mayor estabilidad?",

        "¿Los resultados numericos son consistentes con "
        "los resultados analiticos?"
    ]

    for i, pregunta in enumerate(preguntas, 1):

        st.write(f"**{i}.** {pregunta}")

    # Exportar
    st.divider()

    st.subheader("📥 Descargar resultados")

    excel = crear_excel(df, resumen)

    st.download_button(
        label="Descargar informe Excel",
        data=excel,
        file_name="modelo_rentabilidad.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar datos CSV",
        data=csv,
        file_name="datos_rentabilidad.csv",
        mime="text/csv"
    )

# ================================================================
# 9. INFORMACION FINAL
# ================================================================

st.divider()

st.markdown("""
### Consideraciones metodologicas

1. Los resultados dependen de los parametros introducidos.
2. La convergencia matematica debe justificarse mediante
   criterios analiticos, no solo mediante graficas.
3. Los costos e impuestos se aplican por periodo.
4. La inflacion se utiliza para calcular el poder adquisitivo.
5. La simulacion Monte Carlo representa escenarios hipoteticos.
6. La volatilidad y las probabilidades de perdida no son
   pronosticos ni garantias de rentabilidad.

**Objetivo:** utilizar el modelo como herramienta de experimentacion,
comparacion y discusion de resultados matematicos y financieros.
""")

st.caption(
    "Modelo academico de rentabilidad | Python + Streamlit + Plotly"
)
