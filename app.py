import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Plataforma Analítica | BI, IA e Governança",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 14px 18px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def formatar_brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def gerar_base_corporativa(n_registros: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nomes_base = [
        "Ana Silva", "Bruno Costa", "Carlos Mendes", "Daniela Rocha", "Eduardo Lima",
        "Fernanda Alves", "Gabriel Souza", "Helena Martins", "Igor Ferreira", "Juliana Dias",
        "Lucas Pereira", "Mariana Gomes", "Nicolas Barbosa", "Olivia Ribeiro", "Pedro Carvalho",
        "Rafaela Nunes", "Samuel Castro", "Tatiana Moreira", "Vinicius Araujo", "Yasmin Cardoso",
    ]
    nomes = [f"{rng.choice(nomes_base)} #{i:03d}" for i in range(1, n_registros + 1)]
    cpfs = [
        f"{rng.integers(100,999)}.{rng.integers(100,999)}.{rng.integers(100,999)}-{rng.integers(10,99)}"
        for _ in range(n_registros)
    ]
    emails = [
        f"usuario.{i:03d}@{rng.choice(['empresa.com.br', 'corp.io', 'cliente.net'])}"
        for i in range(1, n_registros + 1)
    ]

    regioes = rng.choice(
        ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"],
        size=n_registros,
        p=[0.42, 0.24, 0.18, 0.10, 0.06],
    )
    segmentos = rng.choice(
        ["Enterprise", "Mid-Market", "SMB"],
        size=n_registros,
        p=[0.20, 0.45, 0.35],
    )
    idades = rng.integers(21, 68, size=n_registros)

    base_renda = np.where(
        segmentos == "Enterprise", 14500, np.where(segmentos == "Mid-Market", 7800, 3900)
    )
    renda = np.round(rng.normal(base_renda, base_renda * 0.25).clip(1800, 32000), 2)
    ticket_medio = np.round(renda * rng.uniform(0.08, 0.22, size=n_registros), 2)
    dias_inativo = rng.integers(1, 150, size=n_registros)
    chamados_suporte = rng.poisson(lam=2.1, size=n_registros)

    logit = (
        -2.4
        + 0.028 * dias_inativo
        + 0.35 * chamados_suporte
        - 0.00006 * renda
        + np.where(regioes == "Nordeste", 0.35, 0.0)
    )
    prob_churn = np.round((1 / (1 + np.exp(-logit))) * 100, 1)

    return pd.DataFrame(
        {
            "id_cliente": range(1, n_registros + 1),
            "nome": nomes,
            "cpf": cpfs,
            "email": emails,
            "regiao": regioes,
            "segmento": segmentos,
            "idade": idades,
            "renda_mensal": renda,
            "ticket_medio": ticket_medio,
            "dias_inativo": dias_inativo,
            "chamados_suporte": chamados_suporte,
            "prob_churn": prob_churn,
        }
    )


# Sidebar
with st.sidebar:
    st.header("Configuração do Pipeline")
    fonte_dados = st.radio(
        "Origem da Base de Dados",
        ["Base Sintética (Varejo B2B/B2C)", "Upload de Arquivo CSV"],
    )

    if fonte_dados == "Upload de Arquivo CSV":
        arquivo = st.file_uploader("Selecione um CSV estruturado", type=["csv"])
        if arquivo is not None:
            df = pd.read_csv(arquivo)
            st.success(f"Base carregada: {len(df)} registros.")
        else:
            st.info("Aguardando upload. Usando base padrão temporariamente.")
            df = gerar_base_corporativa()
    else:
        tamanho = st.slider("Volume de registros simulados", 300, 2500, 600, step=100)
        seed = st.number_input("Semente aleatória (Seed)", value=42, step=1)
        df = gerar_base_corporativa(tamanho, int(seed))

    st.divider()
    st.caption("Filtros Globais (Camada Semântica)")
    if "regiao" in df.columns:
        regioes_sel = st.multiselect(
            "Região",
            options=sorted(df["regiao"].unique()),
            default=sorted(df["regiao"].unique()),
        )
        df_filtrado = df[df["regiao"].isin(regioes_sel)].copy()
    else:
        df_filtrado = df.copy()

# Cabeçalho Principal
st.title("Arquitetura Analítica: Tendências, IA Aplicada e Privacidade")
st.caption(
    "Ambiente interativo de demonstração de Real-Time BI, Augmented Analytics e conformidade LGPD."
)

if df_filtrado.empty:
    st.warning("Selecione pelo menos uma região na barra lateral para exibir os indicadores.")
    st.stop()

# KPIs Executivos
if {"ticket_medio", "prob_churn", "renda_mensal"}.issubset(df_filtrado.columns):
    k1, k2, k3, k4 = st.columns(4)
    receita_total = df_filtrado["ticket_medio"].sum()
    receita_risco = df_filtrado.loc[df_filtrado["prob_churn"] >= 65, "ticket_medio"].sum()
    churn_medio = df_filtrado["prob_churn"].mean()

    k1.metric("Registros Ativos", f"{len(df_filtrado):,}".replace(",", "."))
    k2.metric("Receita Recorrente (MRR)", formatar_brl(receita_total))
    k3.metric("Risco Médio de Churn", f"{churn_medio:.1f}%")
    k4.metric("Receita sob Risco Crítico", formatar_brl(receita_risco))

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "Streaming & Detecção de Anomalias",
        "Augmented Analytics & NLQ",
        "Governança, LGPD & K-Anonymity",
    ]
)

# ==============================================================================
# TAB 1: TENDÊNCIAS FUTURAS (STREAMING & CONTROLE ESTATÍSTICO DE PROCESSO)
# ==============================================================================
with tab1:
    st.subheader("Monitoramento Contínuo e Detecção Estatística de Anomalias")
    st.write(
        "Comparação entre séries temporais em tempo real e bandas de controle dinâmicas (Z-Score) para identificação proativa de desvios operacionais."
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        z_threshold = st.slider("Limiar Z-Score (Desvios Padrão)", 1.5, 3.5, 2.3, 0.1)
        janela_mm = st.slider("Janela da Média Móvel (períodos)", 3, 12, 5)
        ruido_seed = st.number_input("Lote de Telemetria #", min_value=1, max_value=999, value=10)

    rng_stream = np.random.default_rng(int(ruido_seed))
    timestamps = pd.date_range("2026-11-16 08:00", periods=48, freq="15min")
    tendencia = np.linspace(480, 560, 48)
    sazonalidade = 35 * np.sin(np.linspace(0, 3 * np.pi, 48))
    valores = tendencia + sazonalidade + rng_stream.normal(0, 22, 48)

    valores[16] += 145
    valores[37] -= 135

    df_ts = pd.DataFrame({"timestamp": timestamps, "volume_eventos": valores})
    df_ts["media_movel"] = df_ts["volume_eventos"].rolling(janela_mm, min_periods=1).mean()
    df_ts["desvio_movel"] = df_ts["volume_eventos"].rolling(janela_mm, min_periods=1).std().fillna(20)
    df_ts["banda_sup"] = df_ts["media_movel"] + z_threshold * df_ts["desvio_movel"]
    df_ts["banda_inf"] = df_ts["media_movel"] - z_threshold * df_ts["desvio_movel"]
    df_ts["anomalia"] = (df_ts["volume_eventos"] > df_ts["banda_sup"]) | (
        df_ts["volume_eventos"] < df_ts["banda_inf"]
    )

    with c2:
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["timestamp"],
                y=df_ts["banda_sup"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["timestamp"],
                y=df_ts["banda_inf"],
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.15)",
                line=dict(width=0),
                name="Banda de Normalidade",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df_ts["timestamp"],
                y=df_ts["volume_eventos"],
                mode="lines",
                name="Eventos / 15min",
                line=dict(color="#2563EB", width=2),
            )
        )
        anomalias_df = df_ts[df_ts["anomalia"]]
        fig_ts.add_trace(
            go.Scatter(
                x=anomalias_df["timestamp"],
                y=anomalias_df["volume_eventos"],
                mode="markers",
                name="Anomalia Detectada",
                marker=dict(color="#DC2626", size=10, symbol="diamond"),
            )
        )
        fig_ts.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.08),
            xaxis_title="Janela Temporal",
            yaxis_title="Volume de Eventos",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        if not anomalias_df.empty:
            horarios = ", ".join(anomalias_df["timestamp"].dt.strftime("%H:%M"))
            st.warning(f"Alertas disparados automaticamente nos horários: {horarios}")

# ==============================================================================
# TAB 2: AUGMENTED ANALYTICS, NLQ & SIMULADOR PRESCRITIVO
# ==============================================================================
with tab2:
    st.subheader("Consulta em Linguagem Natural (NLQ) e Modelagem Prescritiva")

    col_nlq, col_sim = st.columns([3, 2], gap="large")

    with col_nlq:
        st.markdown("**Interface Conversacional (NLQ)**")
        consulta_texto = st.text_input(
            "Digite uma pergunta sobre a base (ex: 'churn por regiao', 'ticket por segmento', 'inatividade vs churn'):",
            value="churn por regiao",
        ).lower()

        if {"regiao", "segmento", "prob_churn", "ticket_medio", "dias_inativo"}.issubset(df_filtrado.columns):
            if "segmento" in consulta_texto:
                agg = df_filtrado.groupby("segmento", as_index=False)[["ticket_medio", "prob_churn"]].mean()
                fig = px.bar(
                    agg,
                    x="segmento",
                    y="ticket_medio",
                    color="prob_churn",
                    color_continuous_scale="Blues",
                    labels={
                        "segmento": "Segmento",
                        "ticket_medio": "Ticket Médio (R$)",
                        "prob_churn": "Risco Churn (%)",
                    },
                    title="Ticket Médio e Risco de Churn por Segmento",
                )
                st.plotly_chart(fig, use_container_width=True)
                melhor = agg.sort_values("ticket_medio", ascending=False).iloc[0]
                st.info(
                    f"Diagnóstico Automatizado: O segmento {melhor['segmento']} concentra o maior ticket médio ({formatar_brl(melhor['ticket_medio'])})."
                )

            elif "inatividade" in consulta_texto or "dias" in consulta_texto or "correlacao" in consulta_texto:
                fig = px.scatter(
                    df_filtrado,
                    x="dias_inativo",
                    y="prob_churn",
                    color="segmento",
                    opacity=0.75,
                    labels={
                        "dias_inativo": "Dias sem Interação",
                        "prob_churn": "Probabilidade de Churn (%)",
                    },
                    title="Dispersão: Inatividade vs. Risco Preditivo de Churn",
                )
                st.plotly_chart(fig, use_container_width=True)
                corr = df_filtrado["dias_inativo"].corr(df_filtrado["prob_churn"])
                st.info(
                    f"Diagnóstico Automatizado: Correlação linear de Pearson r = {corr:.2f} entre dias de inatividade e risco de cancelamento."
                )

            else:
                agg = (
                    df_filtrado.groupby("regiao", as_index=False)["prob_churn"]
                    .mean()
                    .sort_values("prob_churn", ascending=False)
                )
                fig = px.bar(
                    agg,
                    x="regiao",
                    y="prob_churn",
                    labels={
                        "regiao": "Região",
                        "prob_churn": "Probabilidade Média de Churn (%)",
                    },
                    title="Risco Médio de Churn por Região",
                    color_discrete_sequence=["#1E40AF"],
                )
                st.plotly_chart(fig, use_container_width=True)
                top_reg = agg.iloc[0]
                st.info(
                    f"Diagnóstico Automatizado: A região {top_reg['regiao']} apresenta o maior risco médio de churn ({top_reg['prob_churn']:.1f}%)."
                )

    with col_sim:
        st.markdown("**Simulador Prescritivo (What-If Analysis)**")
        st.write("Simule o impacto financeiro de uma campanha de retenção sobre clientes de alto risco.")
        limiar_risco = st.slider("Limiar de Risco para Intervenção (%)", 50, 90, 70)
        taxa_conversao = st.slider("Taxa Esperada de Retenção (%)", 10, 60, 30)
        custo_incentivo = st.number_input("Custo Médio por Cliente Abordado (R$)", 20.0, 500.0, 85.0, step=5.0)

        if {"prob_churn", "ticket_medio"}.issubset(df_filtrado.columns):
            alvo = df_filtrado[df_filtrado["prob_churn"] >= limiar_risco]
            qtd_alvo = len(alvo)
            receita_preservada = alvo["ticket_medio"].sum() * (taxa_conversao / 100) * 6
            custo_total = qtd_alvo * custo_incentivo
            roi = ((receita_preservada - custo_total) / custo_total * 100) if custo_total > 0 else 0.0

            st.metric("Clientes Elegíveis à Ação", f"{qtd_alvo}")
            st.metric("Receita Preservada (6 meses)", formatar_brl(receita_preservada))
            st.metric("ROI Projetado da Intervenção", f"{roi:.1f}%")

# ==============================================================================
# TAB 3: PRIVACIDADE, LGPD & AUDITORIA DE K-ANONYMITY
# ==============================================================================
with tab3:
    st.subheader("Privacidade por Design (LGPD) e Auditoria de Reidentificação")

    col_ctrl, col_audit = st.columns([2, 1])
    with col_ctrl:
        modo_protecao = st.selectbox(
            "Política de Governança Aplicada à Camada de Consumo (BI):",
            [
                "Nível 0 — Dados Brutos (Acesso Restrito / Violação LGPD em BI)",
                "Nível 1 — Mascaramento Dinâmico (Data Masking)",
                "Nível 2 — Pseudonimização Criptográfica (SHA-256 + Salt)",
                "Nível 3 — Anonimização por Generalização (K-Anonymity)",
            ],
        )

    colunas_exibicao = [
        c
        for c in ["id_cliente", "nome", "cpf", "email", "regiao", "idade", "renda_mensal", "prob_churn"]
        if c in df_filtrado.columns
    ]
    df_gov = df_filtrado[colunas_exibicao].copy()

    if "Nível 0" in modo_protecao:
        risco_label = "Crítico (PII Exposto)"
        k_estimado = 1
        st.error("Identificadores diretos (Nome, CPF, E-mail) visíveis em texto claro.")

    elif "Nível 1" in modo_protecao:
        risco_label = "Alto (Reidentificável por cruzamento)"
        k_estimado = 1
        if "nome" in df_gov.columns:
            df_gov["nome"] = df_gov["nome"].str.slice(0, 3) + "****"
        if "cpf" in df_gov.columns:
            df_gov["cpf"] = "***.***." + df_gov["cpf"].str.slice(8, 11) + "-**"
        if "email" in df_gov.columns:
            df_gov["email"] = df_gov["email"].str.replace(r"(^.).*(@.*$)", r"\1***\2", regex=True)

    elif "Nível 2" in modo_protecao:
        risco_label = "Moderado (LGPD Art. 13 — Dado Pseudonimizado)"
        salt = "UNISO_2026_SALT"
        for col in ["nome", "cpf", "email"]:
            if col in df_gov.columns:
                df_gov[col] = df_gov[col].apply(
                    lambda v: hashlib.sha256(f"{salt}:{v}".encode()).hexdigest()[:14]
                )
        qi_cols = [c for c in ["regiao", "idade"] if c in df_gov.columns]
        k_estimado = int(df_gov.groupby(qi_cols, observed=True).size().min()) if qi_cols else 1

    else:
        risco_label = "Baixo (Dado Anonimizado — Fora do escopo restritivo)"
        df_gov = df_gov.drop(columns=[c for c in ["id_cliente", "nome", "cpf", "email"] if c in df_gov.columns])
        if "idade" in df_gov.columns:
            df_gov["faixa_etaria"] = pd.cut(
                df_gov["idade"], bins=[18, 35, 50, 75], labels=["19-35", "36-50", "51+"]
            )
            df_gov = df_gov.drop(columns=["idade"])
        if "renda_mensal" in df_gov.columns:
            df_gov["faixa_renda"] = pd.cut(
                df_gov["renda_mensal"],
                bins=[0, 5000, 10000, 50000],
                labels=["Até R$ 5k", "R$ 5k–10k", "Acima de R$ 10k"],
            )
            df_gov = df_gov.drop(columns=["renda_mensal"])
        qi_cols = [c for c in ["regiao", "faixa_etaria", "faixa_renda"] if c in df_gov.columns]
        k_estimado = int(df_gov.groupby(qi_cols, observed=True).size().min()) if qi_cols else 1

    with col_audit:
        st.metric(
            "Índice K-Anonymity Mínimo",
            f"k = {k_estimado}",
            help="Menor tamanho de grupo indistinguível considerando os quase-identificadores.",
        )

    st.caption(f"Classificação de Risco de Privacidade: **{risco_label}**")
    st.dataframe(df_gov.head(15), use_container_width=True, hide_index=True)
