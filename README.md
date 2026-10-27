# Mortalidade no trânsito no Estado de São Paulo · Dashboard (Checkpoint 3)

Data Science & Analytics · FIAP · 2º semestre de 2026

| Integrante | RM |
|---|---|
| Rogério Deligi Ferreira Filho | RM561942 |
| Maria Fernanda Garavelli Dantas | RM562686 |

**Aplicação publicada (acesso público, sem instalação):** https://rogeriooxy.github.io/fiap-dsa-cp3-transito-sp/

A dashboard reúne os indicadores da Parte 1 (taxa de óbitos por 100 mil habitantes, meta da Resolução CONTRAN 1.004/2023,
participação da motocicleta, recortes por região, turno, tipo de via e município) e os resultados da análise preditiva da
Parte 2 (três modelos que estimam se um sinistro é fatal, comparados por ROC-AUC, PR-AUC e recall no top 10%, com simulador
de cenários).

## Fonte dos dados

| Fonte | Conteúdo | Licença |
|---|---|---|
| [Infosiga SP – Detran-SP](https://dadosabertos.sp.gov.br/dataset/sinistros-infosiga) | 1.422.535 sinistros, jan/2015 a jul/2026 | CC-BY 4.0 |
| [IBGE – API de Agregados v3](https://servicodados.ibge.gov.br/api/docs/agregados?versao=3), agregados 6579 e 4714 | população por município e ano | dados públicos |
| GeoJSON dos municípios de SP ([tbrugz/geodata-br](https://github.com/tbrugz/geodata-br)) | contorno dos 645 municípios | domínio público |

Dados acessados entre 26 e 29/08/2026. O repositório traz apenas as tabelas agregadas que a aplicação usa (`app_dados/`);
a base completa (62 MB em parquet) é gerada a partir dos CSVs mensais do Infosiga, como descrito no notebook da Parte 1.

## Estrutura

```
app.py                      aplicação Streamlit (a dashboard)
app_dados/                  tabelas agregadas, métricas e curvas dos modelos, GeoJSON
assets/                     gráficos estáticos (matplotlib) gerados pelo notebook
CP03_DSA_Dashboard_...ipynb notebook executado: preparação dos dados, KPIs e os 3 modelos
pipeline_cp3.py             o mesmo pipeline em formato de script (células # %%)
index.html                  versão publicada: a mesma app.py rodando no navegador com stlite
requirements.txt            dependências para rodar localmente
```

## Como executar localmente

Requisitos: Python 3.10 ou superior.

```bash
git clone https://github.com/RogerioOxy/fiap-dsa-cp3-transito-sp.git
cd fiap-dsa-cp3-transito-sp
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abre em `http://localhost:8501`. Ela só lê os arquivos de `app_dados/` e `assets/`, que já estão no repositório.

Para regenerar as tabelas e os modelos é preciso a base consolidada da Parte 1 (`infosiga_sinistros_2015_2026.parquet` e
`ibge_populacao_municipal_sp.parquet`); ajuste a variável `DADOS` no início do notebook e execute todas as células.

## Como a versão publicada funciona

A página do GitHub Pages carrega o [stlite](https://github.com/whitphx/stlite), que executa o Streamlit dentro do navegador
(Pyodide/WebAssembly). É o mesmo `app.py`, sem servidor: a primeira abertura leva de 20 a 60 segundos para baixar o Python
e as bibliotecas; depois disso os filtros respondem na hora.

## Modelos (resumo)

Classificação binária do sinistro (fatal ou não fatal), sinistros de 2019 em diante, sem as variáveis que entregam o desfecho
(tipo de registro, gravidade das vítimas e os campos "não disponível" que só aparecem no cadastro de sinistro fatal).
Treino com 250 mil linhas e teste com 185 mil linhas nunca vistas no treino.

| Modelo | Biblioteca | ROC-AUC | PR-AUC | Recall no top 10% |
|---|---|---|---|---|
| Regressão logística | scikit-learn | 0,894 | 0,340 | 57,5% |
| LightGBM | lightgbm | 0,933 | 0,549 | 72,4% |
| Rede neural MLP (128-64-32) | scikit-learn | 0,933 | 0,553 | 72,3% |
