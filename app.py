import streamlit as st
import pandas as pd
import numpy as np
import hashlib

# Configuração da página
st.set_page_config(
    page_title="BI & Big Data Analytics - UNISO",
    page_icon="📊",
    layout="wide"
)

# Cabeçalho
st.title("📊 BI & Big Data: Tendências, Inteligência Artificial e Privacidade")
st.markdown("**Demonstração Prática — Apresentação Acadêmica UNISO**")
st.divider()

# Gerando base de dados sintética para a demonstração
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n = 150
    nomes = [f"Cliente_{i:03d}" for i in range(1, n + 1)]
    cpfs = [f"{np.random.randint(100,999)}.{np.random.randint(100,999)}.{np.random.randint(100,999)}-{np.random.randint(10,99)}" for _ in range(n)]
    regioes = np.random.choice(["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"], n, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    idades = np.random.randint(19, 72, n)
    renda = np.round(np.random.normal(5200, 2100, n).clip(1500, 18000), 2)
    ticket_medio = np.round(renda * np.random.uniform(0.05, 0.18, n), 2)
    dias_sem_compra = np.random.randint(1, 120, n)
    
    # Probabilidade de Churn (IA Preditiva simulada)
    prob_churn = np.clip((dias_sem_compra / 120) * 0.7 + (1 - (renda / 18000)) * 0.3, 0.05, 0.98)
    
    df = pd.DataFrame({
        "ID": range(1, n + 1),
        "Nome": nomes,
        "CPF": cpfs,
        "Regiao": regioes,
        "Idade": idades,
        "Renda_Mensal": renda,
        "Ticket_Medio": ticket_medio,
        "Dias_Sem_Compra": dias_sem_compra,
        "Risco_Churn": np.round(prob_churn * 100, 1)
    })
    return df

df = carregar_dados()

# Criação das 3 Abas alinhadas aos Pilares da Apresentação
aba1, aba2, aba3 = st.tabs([
    "1️⃣ Tendências: Real-Time & Anomalias",
    "2️⃣ Uso de IA: Augmented Analytics & NLQ",
    "3️⃣ Privacidade: LGPD & Anonimização"
])

# ==============================================================================
# ABA 1: TENDÊNCIAS FUTURAS (REAL-TIME & DETECÇÃO DE ANOMALIAS)
# ==============================================================================
with aba1:
    st.subheader("De Painéis Estáticos para Monitoramento Ativo (Real-Time Analytics)")
    st.write("Simulação de fluxo contínuo de transações com detecção automática de anomalias (desvio padrão > 2σ).")

    col1, col2 = st.columns([1, 3])
    with col1:
        sensibilidade = st.slider("Limiar de Alerta (Desvios Padrão - σ)", 1.5, 3.5, 2.2, 0.1)
        if st.button("🔄 Simular Novo Lote de Streaming"):
            st.rerun()

    # Dados temporais simulados
    horas = pd.date_range("2026-11-16 08:00", periods=30, freq="15min")
    transacoes = np.random.normal(500, 45, 30)
    # Inserindo 2 anomalias propositais
    transacoes[12] = 740  # Pico anômalo
    transacoes[24] = 280  # Queda brusca

    df_stream = pd.DataFrame({"Horario": horas, "Volume_Transacoes": transacoes}).set_index("Horario")
    media = df_stream["Volume_Transacoes"].mean()
    desvio = df_stream["Volume_Transacoes"].std()
    anomalias = df_stream[np.abs(df_stream["Volume_Transacoes"] - media) > sensibilidade * desvio]

    with col2:
        st.line_chart(df_stream)
        if not anomalias.empty:
            st.error(f"🚨 **{len(anomalias)} Anomalia(s) detectada(s) automaticamente!** Horários: {', '.join(anomalias.index.strftime('%H:%M'))}")
        else:
            st.success("✅ Fluxo operando dentro dos parâmetros normais.")

# ==============================================================================
# ABA 2: USO DE IA (AUGMENTED ANALYTICS, NLQ & PRESCRITIVO)
# ==============================================================================
with aba2:
    st.subheader("Augmented Analytics & Natural Language Querying (NLQ)")
    st.write("Permite que gestores façam perguntas em linguagem natural e recebam diagnósticos preditivos e prescritivos.")

    pergunta = st.selectbox(
        "Selecione uma pergunta de negócio em linguagem natural (ou simule uma consulta):",
        [
            "Qual região apresenta o maior risco médio de cancelamento (Churn)?",
            "Qual a relação entre Renda Mensal e Ticket Médio por região?",
            "Quais clientes exigem ação imediata de retenção (Análise Prescritiva)?"
        ]
    )

    st.markdown("---")
    if "maior risco médio" in pergunta:
        resumo = df.groupby("Regiao")["Risco_Churn"].mean().sort_values(ascending=False)
        pior_regiao = resumo.index[0]
        valor_pior = resumo.iloc[0]
        
        st.info(f"🤖 **Insight Gerado por IA:** A região **{pior_regiao}** lidera o risco médio de churn com **{valor_pior:.1f}%**, impulsionada pelo maior tempo médio sem compras. Recomenda-se campanha direcionada de reengajamento.")
        st.bar_chart(resumo)

    elif "relação entre Renda" in pergunta:
        corr = df["Renda_Mensal"].corr(df["Ticket_Medio"])
        st.info(f"🤖 **Insight Gerado por IA:** Existe uma correlação positiva forte (**r = {corr:.2f}**) entre Renda Mensal e Ticket Médio. Clientes das regiões Sudeste e Sul concentram os maiores tickets.")
        st.scatter_chart(df, x="Renda_Mensal", y="Ticket_Medio", color="Regiao")

    elif "ação imediata" in pergunta:
        criticos = df[df["Risco_Churn"] >= 70].sort_values(by="Risco_Churn", ascending=False)
        st.warning(f"🤖 **Recomendação Prescritiva:** Foram identificados **{len(criticos)} clientes** com risco crítico (≥ 70%). **Ação sugerida:** Enviar cupom automático de 15% de desconto para clientes com mais de 90 dias de inatividade.")
        st.dataframe(criticos[["ID", "Regiao", "Ticket_Medio", "Dias_Sem_Compra", "Risco_Churn"]].head(10), use_container_width=True)

# ==============================================================================
# ABA 3: PRIVACIDADE E LGPD (GOVERNANÇA E PROTEÇÃO DE DADOS)
# ==============================================================================
with aba3:
    st.subheader("Privacidade por Design (LGPD) em Pipelines de Big Data")
    st.write("Alterne o nível de proteção para visualizar como os dados são tratados antes de chegarem ao dashboard de BI:")

    nivel_privacidade = st.radio(
        "Técnica de Proteção Aplicada (PETs):",
        [
            "1. Dados Brutos (Sem Proteção - Violação LGPD)",
            "2. Mascaramento Dinâmico (Data Masking)",
            "3. Pseudonimização (Hash SHA-256)",
            "4. Anonimização & Generalização (K-Anonymity)"
        ],
        horizontal=True
    )

    df_priv = df[["ID", "Nome", "CPF", "Regiao", "Idade", "Renda_Mensal", "Risco_Churn"]].copy()

    if "1. Dados Brutos" in nivel_privacidade:
        st.error("⚠️ **ALERTA DE CONFORMIDADE:** Dados pessoais identificáveis (PII) expostos diretamente para analistas de BI.")
    
    elif "2. Mascaramento" in nivel_privacidade:
        st.warning("🛡️ **Mascaramento Aplicado:** Identificadores diretos ocultados parcialmente para consulta operacional.")
        df_priv["Nome"] = df_priv["Nome"].apply(lambda x: x[:3] + "****")
        df_priv["CPF"] = df_priv["CPF"].apply(lambda x: "***." + x[4:7] + ".***-**")

    elif "3. Pseudonimização" in nivel_privacidade:
        st.info("🔐 **Pseudonimização (LGPD Art. 13):** Identificadores substituídos por chaves criptográficas irreversíveis sem a chave secreta, mantendo a rastreabilidade analítica.")
        df_priv["Nome"] = df_priv["Nome"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:12])
        df_priv["CPF"] = df_priv["CPF"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])

    elif "4. Anonimização" in nivel_privacidade:
        st.success("✅ **Dado Anonimizado (Fora do escopo restritivo da LGPD):** Remoção de identificadores diretos e generalização de quase-identificadores em faixas.")
        df_priv = df_priv.drop(columns=["Nome", "CPF"])
        df_priv["Faixa_Etaria"] = pd.cut(df_priv["Idade"], bins=[18, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
        df_priv["Faixa_Renda"] = pd.cut(df_priv["Renda_Mensal"], bins=[0, 3000, 7000, 20000], labels=["Até R$ 3k", "R$ 3k - 7k", "Acima de R$ 7k"])
        df_priv = df_priv.drop(columns=["Idade", "Renda_Mensal"])

    st.dataframe(df_priv.head(12), use_container_width=True)