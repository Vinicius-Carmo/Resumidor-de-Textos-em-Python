import nltk
import string
import numpy as np
import torch
import networkx as nx
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- Garantir recursos NLTK ----------
def garantir_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)

# Pontos importantes:
garantir_nltk_resource('tokenizers/punkt', 'punkt')
garantir_nltk_resource('corpora/stopwords', 'stopwords')
# Algumas versões do NLTK mais recentes usam 'punkt_tab':
garantir_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')

st.write("NLTK pronto!")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Josefin+Sans:ital,wght@0,100..700;1,100..700&family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

def carregar_css(arquivo_css):
    with open(arquivo_css) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css("style.css")

stopwords = nltk.corpus.stopwords.words('portuguese')

#Preprocessamento do texto. Retira Pontuações, Stopwords e deixa tudo em lowercase
def preprocessamento(texto):
    texto_formatado = texto.lower()
    tokens = nltk.word_tokenize(texto_formatado)
    tokens_noStop = [w for w in tokens if w not in stopwords and w not in string.punctuation]
    return ' '.join(tokens_noStop)

#Carregando modelo de sentencas do transformers
@st.cache_resource #Modelo carrega apenas uma vez
def carregar_modelo():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

modelo = carregar_modelo()

#Sumarização do texto, usando similaridade de cosseno no embedding do modelo importado
def sumarizar(texto, percentual_resumo):
    sentencas_originais = nltk.sent_tokenize(texto)
    sentencas_formatadas = [preprocessamento(s) for s in sentencas_originais]

    if len(sentencas_originais) == 0:
        return "Nenhuma sentença válida encontrada no texto."

    embeddings = modelo.encode(sentencas_formatadas)

    matriz_similaridade = cosine_similarity(embeddings)

    grafo_similaridade = nx.from_numpy_array(matriz_similaridade)

    notas = nx.pagerank(grafo_similaridade)
    notas_ordenadas = sorted(((notas[i], s) for i, s in enumerate(sentencas_originais)), reverse=True)

    qtd_sentencas = max(1, int(len(sentencas_originais) * percentual_resumo / 100))
    melhores_sentencas = [notas_ordenadas[i][1] for i in range(qtd_sentencas)]

    resumo = ' '.join(melhores_sentencas)
    return resumo


#Interface Web


st.title('𓄿 CARMO Resumidor de Textos')
st.write("Resumo automático de textos em português com base em embeddings e PageRank.")

texto_usuario = st.text_area(placeholder= 'No mínimo 2 frases...',label="Insira o texto aqui:", height=250)
if "percentual" not in st.session_state:
    st.session_state.percentual = 50

percentual = st.number_input("Escolha o percentual de resumo:", 
                             min_value=1, max_value=100, 
                             value=st.session_state.percentual, step=1)

st.session_state.percentual = percentual

if st.button("Gerar Resumo"):
    if texto_usuario.strip():
        with st.spinner("Gerando resumo... Aguarde alguns segundos."):
            resumo = sumarizar(texto_usuario, percentual)
        st.subheader("Resumo gerado:")
        st.write(resumo)
    else:
        st.warning("Por favor, insira um texto para resumir.")