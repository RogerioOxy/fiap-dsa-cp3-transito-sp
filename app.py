"""Dashboard do Checkpoint 3 de Data Science & Analytics (FIAP, 2º semestre de 2026).

Mortalidade no trânsito no Estado de São Paulo: indicadores da Parte 1 e análise preditiva da Parte 2.
Rogério Deligi Ferreira Filho (RM561942) e Maria Fernanda Garavelli Dantas (RM562686).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path("app_dados")
A = Path("assets")

st.set_page_config(page_title="Trânsito SP | Mortalidade", page_icon="🚦", layout="wide")


@st.cache_data
def carregar():
    cubo = pd.read_csv(D / "cubo.csv")
    pop = pd.read_csv(D / "pop_regiao.csv")
    mun = pd.read_csv(D / "municipios.csv")
    anual = pd.read_csv(D / "anual.csv")
    metricas = pd.read_csv(D / "metricas.csv")
    roc = pd.read_csv(D / "roc.csv")
    pr = pd.read_csv(D / "pr.csv")
    imp = pd.read_csv(D / "importancias.csv")
    cen = pd.read_csv(D / "cenarios.csv")
    resumo = json.loads((D / "resumo.json").read_text(encoding="utf-8"))
    geo = json.loads((D / "municipios_sp.geojson").read_text(encoding="utf-8"))
    return cubo, pop, mun, anual, metricas, roc, pr, imp, cen, resumo, geo


cubo, pop, mun, anual, metricas, roc, pr, imp, cen, resumo, geo = carregar()

st.title("Mortalidade no trânsito no Estado de São Paulo")
st.caption("Infosiga SP (Detran-SP) + população IBGE · jan/2015 a jul/2026 · "
           "Rogério Deligi Ferreira Filho (RM561942) e Maria Fernanda Garavelli Dantas (RM562686)")

# ---------------- Filtros ----------------
st.sidebar.header("Filtros")
anos = st.sidebar.slider("Período", 2015, 2026, (2019, 2025))
regioes = sorted(cubo["regiao"].unique())
sel_reg = st.sidebar.multiselect("Região administrativa", regioes, placeholder="Todas")
vias = sorted(cubo["tipo_via"].unique())
sel_via = st.sidebar.multiselect("Tipo de via", vias, placeholder="Todos")
sel_moto = st.sidebar.radio("Motocicleta envolvida", ["Todos", "Com motocicleta", "Sem motocicleta"])
tipos = sorted(cubo["tp_sinistro_primario"].unique())
sel_tipo = st.sidebar.multiselect("Tipo de sinistro", tipos, placeholder="Todos")
st.sidebar.caption("2026 é parcial (jan a jul). Sinistros não fatais só existem na base a partir de 2019.")

f = cubo[cubo["ano"].between(*anos)]
if sel_reg:
    f = f[f["regiao"].isin(sel_reg)]
if sel_via:
    f = f[f["tipo_via"].isin(sel_via)]
if sel_moto != "Todos":
    f = f[f["moto"] == sel_moto]
if sel_tipo:
    f = f[f["tp_sinistro_primario"].isin(sel_tipo)]

p = pop[pop["ano"].between(*anos)]
if sel_reg:
    p = p[p["regiao"].isin(sel_reg)]
pop_ano = p.groupby("ano")["populacao"].sum()

# ---------------- KPIs ----------------
obitos = int(f["obitos"].sum())
fatais = int(f["sinistros_fatais"].sum())
anos_completos = [a for a in range(anos[0], anos[1] + 1) if a <= 2025]
ob_ano = f.groupby("ano")["obitos"].sum()
taxa = (ob_ano.reindex(anos_completos).sum() / pop_ano.reindex(anos_completos).sum() * 1e5) if anos_completos else float("nan")
moto_fatais = f.loc[f["moto"] == "Com motocicleta", "sinistros_fatais"].sum()
part_moto = moto_fatais / fatais * 100 if fatais else 0
f19 = f[f["ano"] >= 2019]
letal = f19["sinistros_fatais"].sum() / f19["sinistros"].sum() * 100 if f19["sinistros"].sum() else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Óbitos no período", f"{obitos:,}".replace(",", "."))
c2.metric("Sinistros fatais", f"{fatais:,}".replace(",", "."))
c3.metric("Óbitos por 100 mil hab./ano", f"{taxa:.2f}".replace(".", ","), help="Média dos anos completos do período (até 2025)")
c4.metric("Motocicleta nos fatais", f"{part_moto:.1f}%".replace(".", ","))
c5.metric("Letalidade (fatais / sinistros)", f"{letal:.2f}%".replace(".", ",") if letal == letal else "sem dado",
          help="Só a partir de 2019, quando a base passou a registrar os sinistros não fatais")

aba1, aba2, aba3, aba4 = st.tabs(["Visão geral", "Onde e quando", "Modelo preditivo", "Fonte e método"])

# ---------------- Visão geral ----------------
with aba1:
    st.subheader("Meta legal: CONTRAN 1.004/2023")
    m1, m2, m3 = st.columns(3)
    m1.metric("Taxa em 2020 (referência)", f"{resumo['taxa_2020']:.2f}".replace(".", ","))
    m2.metric("Taxa em 2025", f"{resumo['taxa_2025']:.2f}".replace(".", ","),
              delta=f"{resumo['taxa_2025'] - resumo['meta_2025']:+.2f} acima da meta de {resumo['meta_2025']:.2f}".replace(".", ","),
              delta_color="inverse")
    m3.metric("Vidas acima da meta em 2025", f"{resumo['vidas_acima_meta_2025']:,.0f}".replace(",", "."))
    st.caption("Indicadores do Estado inteiro, sem os filtros. A meta reduz 5% do índice de 2020 por ano, até a metade em 2030.")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Série histórica da taxa (gráfico estático)**")
        st.image(str(A / "serie_taxa.png"))
    with g2:
        st.markdown("**Participação da motocicleta nos fatais (gráfico estático)**")
        st.image(str(A / "moto_participacao.png"))

    st.markdown("**Óbitos por ano com os filtros aplicados (interativo)**")
    serie = f.groupby(["ano", "moto"], as_index=False)["obitos"].sum()
    fig = px.bar(serie, x="ano", y="obitos", color="moto", labels={"ano": "Ano", "obitos": "Óbitos", "moto": ""},
                 color_discrete_map={"Com motocicleta": "#e8833a", "Sem motocicleta": "#355c7d"})
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)

# ---------------- Onde e quando ----------------
with aba2:
    col1, col2 = st.columns(2)
    with col1:
        reg = f.groupby("regiao", as_index=False)["obitos"].sum()
        reg = reg.merge(p.groupby("regiao", as_index=False)["populacao"].sum(), on="regiao")
        reg["taxa"] = reg["obitos"] / reg["populacao"] * 1e5
        reg = reg.sort_values("taxa")
        fig = px.bar(reg, x="taxa", y="regiao", orientation="h", labels={"taxa": "Óbitos por 100 mil hab. (média anual)", "regiao": ""},
                     hover_data={"obitos": True}, color="taxa", color_continuous_scale="Reds")
        fig.update_layout(height=520, coloraxis_showscale=False, title="Taxa por região administrativa")
        st.plotly_chart(fig)
    with col2:
        turno = f[f["turno"] != "NAO DISPONIVEL"].groupby("turno", as_index=False)[["obitos", "sinistros_fatais", "sinistros"]].sum()
        ordem = ["MADRUGADA", "MANHA", "TARDE", "NOITE"]
        fig = px.bar(turno, x="turno", y="obitos", category_orders={"turno": ordem}, labels={"turno": "Turno", "obitos": "Óbitos"},
                     color_discrete_sequence=["#6c5b7b"], title="Óbitos por turno")
        fig.update_layout(height=250)
        st.plotly_chart(fig)
        tipo = f.groupby("tp_sinistro_primario", as_index=False)["obitos"].sum().sort_values("obitos")
        fig = px.bar(tipo, x="obitos", y="tp_sinistro_primario", orientation="h", title="Óbitos por tipo de sinistro",
                     labels={"obitos": "Óbitos", "tp_sinistro_primario": ""}, color_discrete_sequence=["#c06c84"])
        fig.update_layout(height=250)
        st.plotly_chart(fig)

    st.markdown("**Mapa: óbitos por 100 mil habitantes por município**")
    ano_mapa = st.selectbox("Ano do mapa", list(range(2025, 2014, -1)))
    mm = mun[mun["ano"] == ano_mapa].copy()
    if sel_reg:
        mm = mm[mm["regiao"].isin(sel_reg)]
    mm["taxa"] = mm["obitos"] / mm["populacao"] * 1e5
    mm["id"] = mm["cod_ibge"].astype(str)
    fig = px.choropleth(mm, geojson=geo, locations="id", featureidkey="properties.id", color="taxa",
                        color_continuous_scale="Reds", range_color=(0, float(mm["taxa"].quantile(0.95)) or 1),
                        hover_name="municipio", hover_data={"id": False, "obitos": True, "populacao": True, "taxa": ":.1f"},
                        labels={"taxa": "por 100 mil"})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.caption("Município pequeno oscila muito de um ano para o outro: poucos óbitos já mudam a taxa. Use o ranking com cautela.")
    rk = mm[mm["populacao"] >= 50_000].sort_values("taxa", ascending=False).head(15)
    st.dataframe(rk[["municipio", "regiao", "populacao", "obitos", "taxa"]].round(2), hide_index=True)
    st.caption("Ranking: os 15 municípios com mais de 50 mil habitantes e maior taxa no ano escolhido.")

# ---------------- Modelo preditivo ----------------
with aba3:
    st.subheader("Qual sinistro tende a ser fatal? (Parte 2)")
    st.markdown(
        f"Classificação binária do sinistro em **fatal / não fatal**, com {resumo['base_modelagem_linhas']:,} sinistros de 2019 em diante "
        f"({resumo['base_modelagem_taxa_fatal']:.2%} fatais). Treino com {resumo['treino_linhas']:,} linhas e teste com "
        f"{resumo['teste_linhas']:,} linhas nunca vistas no treino. Variáveis que entregam o desfecho foram retiradas antes "
        "do treino (tipo de registro, gravidade das vítimas e os campos \"não disponível\" que só aparecem no cadastro de sinistro fatal)."
        .replace(",", "."))
    tabela = metricas[["modelo", "roc_auc", "pr_auc", "recall_top10", "precisao", "recall", "f1", "tempo_treino_s"]].copy()
    tabela.columns = ["Modelo", "ROC-AUC", "PR-AUC", "Recall no top 10%", "Precisão", "Recall", "F1", "Treino (s)"]
    st.dataframe(tabela.round(3), hide_index=True)
    melhor = metricas.iloc[0]
    st.info(f"Melhor modelo pela PR-AUC: **{melhor['modelo']}**. Entre os 10% de sinistros que ele aponta como mais arriscados "
            f"estão **{melhor['recall_top10']:.0%}** de todos os sinistros fatais do conjunto de teste.")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.line(roc, x="fpr", y="tpr", color="modelo", labels={"fpr": "Taxa de falso positivo", "tpr": "Taxa de verdadeiro positivo", "modelo": ""},
                      title="Curva ROC")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Aleatório"))
        fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)
    with g2:
        fig = px.line(pr, x="recall", y="precisao", color="modelo", labels={"recall": "Recall", "precisao": "Precisão", "modelo": ""},
                      title="Curva precisão x recall")
        fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)

    g3, g4 = st.columns(2)
    with g3:
        fig = px.bar(imp.sort_values("importancia"), x="importancia", y="variavel", orientation="h",
                     labels={"importancia": "% do ganho total", "variavel": ""}, title="O que mais pesa no LightGBM",
                     color_discrete_sequence=["#355c7d"])
        fig.update_layout(height=430)
        st.plotly_chart(fig)
    with g4:
        st.markdown("**Comparação dos modelos (gráfico estático)**")
        st.image(str(A / "comparacao_modelos.png"))

    st.markdown("#### Simulador de cenário")
    s1, s2, s3, s4 = st.columns(4)
    via = s1.selectbox("Tipo de via", sorted(cen["tipo_via"].unique()))
    tur = s2.selectbox("Turno", sorted(cen["turno"].unique()))
    tp = s3.selectbox("Tipo de sinistro", sorted(cen["tp_sinistro_primario"].unique()))
    mt = s4.selectbox("Motocicleta", sorted(cen["moto"].unique()))
    linha = cen[(cen["tipo_via"] == via) & (cen["turno"] == tur) & (cen["tp_sinistro_primario"] == tp) & (cen["moto"] == mt)]
    geral = resumo["base_modelagem_taxa_fatal"]
    if linha.empty:
        st.warning("Combinação com menos de 30 sinistros no teste: amostra pequena demais para mostrar.")
    else:
        r = linha.iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Risco médio previsto (LightGBM)", f"{r['risco_previsto']:.1%}".replace(".", ","))
        k2.metric("Letalidade observada no teste", f"{r['letalidade_observada']:.1%}".replace(".", ","),
                  delta=f"{r['letalidade_observada'] / geral:.1f}x a média geral".replace(".", ","), delta_color="off")
        k3.metric("Sinistros nesse cenário (teste)", f"{int(r['sinistros']):,}".replace(",", "."))
        st.caption("O modelo foi treinado com classes balanceadas, então o risco previsto serve para comparar cenários entre si "
                   "(ordenação), não como probabilidade calibrada. A letalidade observada é a proporção real de fatais no teste.")

    st.markdown("#### Como o modelo complementa os indicadores")
    st.markdown(
        "- Os KPIs mostram **quanto** se morre e **onde/quando**; o modelo mostra o **peso de cada fator com os demais controlados** "
        "(por exemplo, separa o efeito da motocicleta do efeito da rodovia).\n"
        "- O recall no top 10% transforma o diagnóstico em **prioridade de ação**: concentrar fiscalização e engenharia de via nos "
        "cenários de maior risco cobre a maior parte das mortes.\n"
        "- Limitações: o boletim não traz velocidade, capacete, álcool nem tempo de socorro; a mudança de critério de registro em "
        "2025 altera a proporção entre classes.")

# ---------------- Fonte e método ----------------
with aba4:
    st.subheader("Fonte dos dados")
    st.markdown(
        "| Fonte | O que traz | Licença |\n|---|---|---|\n"
        "| [Infosiga SP – Detran-SP](https://dadosabertos.sp.gov.br/dataset/sinistros-infosiga) | 1.422.535 sinistros, jan/2015 a jul/2026 | CC-BY 4.0 |\n"
        "| [IBGE – API de Agregados v3](https://servicodados.ibge.gov.br/api/docs/agregados?versao=3) (6579 e 4714) | população por município e ano | dados públicos |\n"
        "| GeoJSON dos municípios de SP (tbrugz/geodata-br) | contorno dos 645 municípios | domínio público |")
    st.markdown(
        "**Cuidados herdados da Parte 1**\n"
        "- Até 2018 a base só registra sinistros fatais: letalidade e qualquer indicador com não fatais começam em 2019.\n"
        "- Em 2022 a população passa a vir do Censo (menor que a estimativa): parte do salto da taxa nesse ano é do denominador.\n"
        "- Em 2025 muda o critério entre notificação e sinistro não fatal; os óbitos não são afetados.\n"
        "- Integração por código IBGE (645 de 645 municípios), nunca por nome.")
    st.markdown("**Integrantes:** Rogério Deligi Ferreira Filho (RM561942) e Maria Fernanda Garavelli Dantas (RM562686) · FIAP 2026")
    st.caption("Dados acessados entre 26 e 29/08/2026. Os dados exibidos são agregados gerados pelo notebook do Checkpoint 3.")
