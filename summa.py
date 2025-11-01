import nltk
import string
import numpy as np
import torch
import networkx as nx
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from io import StringIO
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract
from gtts import gTTS

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------- Recursos NLTK --------------------
def nltk_recursos(recurso, download):
    try:
        nltk.data.find(recurso)
    except LookupError:
        nltk.download(download, quiet=True)

nltk_recursos('tokenizers/punkt', 'punkt')
nltk_recursos('corpora/stopwords', 'stopwords')
nltk_recursos('tokenizers/punkt_tab', 'punkt_tab')


# -------------------- Fontes Google --------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Josefin+Sans:ital,wght@0,100..700;1,100..700&family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# -------------------- Estilo Geral --------------------
st.markdown("""
    <style>
        body {
            background-color: #0c0c0c;
            color: white;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0c0c0c;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stToolbar"] {
            right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


def carregar_css(arquivo_css):
    with open(arquivo_css) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


carregar_css("style.css")


# -------------------- Stopwords --------------------
stopwords = nltk.corpus.stopwords.words('portuguese')


# -------------------- Pré-processamento --------------------
def preprocessamento(texto):
    texto_formatado = texto.lower()
    tokens = nltk.word_tokenize(texto_formatado)
    tokens_noStop = [w for w in tokens if w not in stopwords and w not in string.punctuation]
    return ' '.join(tokens_noStop)


# -------------------- Modelo de Embeddings --------------------
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


modelo = carregar_modelo()


# -------------------- Função de Sumarização --------------------
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


# -------------------- Interface Web ------------------
st.title('𓅂 CARMO Resumo de textos')
st.write("Resumo automático de textos em português com base em embeddings e PageRank.")

uploaded_file = st.file_uploader(
    'Envie um arquivo - (txt, pdf, docx, jpg, jpeg, png):',
    type=['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png']
)

texto_arquivo = ''

if uploaded_file is not None:
    if uploaded_file.type == 'text/plain':
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
        texto_arquivo = stringio.read()

    elif uploaded_file.type == 'application/pdf':
        pdf_reader = PdfReader(uploaded_file)
        texto_arquivo = ' '.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                'application/msword']:
        doc = docx.Document(uploaded_file)
        texto_arquivo = '\n'.join([p.text for p in doc.paragraphs])

    elif uploaded_file.type.startswith('image/'):
        img = Image.open(uploaded_file)
        st.image(img, caption='Imagem enviada com sucesso', use_container_width=True)
        texto_arquivo = pytesseract.image_to_string(img, lang='por')

texto_usuario = st.text_area(
    placeholder='No mínimo 2 frases...',
    value=texto_arquivo,
    label="Digite um texto aqui:",
    height=250
)

if "percentual" not in st.session_state:
    st.session_state.percentual = 50

percentual = st.number_input(
    "Escolha o percentual de resumo:",
    min_value=1, max_value=100,
    value=st.session_state.percentual, step=1
)

st.session_state.percentual = percentual

if st.button("Gerar Resumo"):
    if texto_usuario.strip():
        st.subheader("Resumo gerado:")
        with st.spinner("Gerando... Aguarde alguns segundos."):
            resumo = sumarizar(texto_usuario, percentual)
            tts = gTTS(resumo, lang = 'pt')
            tts.save('audio.mp3')
            audio_arquivo = open('audio.mp3', 'rb')
            audioRead = audio_arquivo.read()
            st.audio(audioRead, format = 'audio/mp3')

        st.write(resumo)
    else:
        st.warning("Por favor, insira um texto para resumir.")
