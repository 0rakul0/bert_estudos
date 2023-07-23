"""

modelagem de topicos tem a vantagem de auto-organizar documentos diversos

"""
import warnings

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

import glob

### Processamento dos dados
def processamento():
    dados = []
    dir = r"./documentos/"
    for arq in glob.iglob(dir+'*.txt',recursive=False):
        meu_arquivo = open(arq, "r", encoding="utf8")
        conteudo = meu_arquivo.read()
        dados.append(conteudo)
        meu_arquivo.close()
    print(f"a quantidade de itens {len(dados)}")
    return dados

def sentencas(dados):
    sentencas = []
    for item in dados:
        for t in sent_tokenize(item):
            sentencas.append(t)
    return sentencas

def modelo():
    modelo = BERTopic(language="portuguese", verbose=True, embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens",
                      top_n_words=15, min_topic_size=10, nr_topics=20,
                      vectorizer_model=CountVectorizer(ngram_range=(1, 3),
                                                       stop_words=stopwords.words("portuguese"), min_df=10))
    return modelo

def treino(modelo, sentencas):
    topicos, probabilidade = modelo.fit_transform(sentencas)

    frequencia = modelo.get_topic_freq()

    fig_visualize_barchart = frequencia.visualize_barchart(n_words=15, top_n_topics=10)
    fig_visualize_hierarchy = frequencia.visualize_hierarchy(top_n_topics=15)
    fig_visualize_heatmap = frequencia.visualize_heatmap(n_clusters=15)

    fig_visualize_barchart.write_html("./visualize_barchart.html")
    fig_visualize_hierarchy.write_html("./fig_visualize_hierarchy.html")
    fig_visualize_heatmap.write_html("./visualize_heatmap.html")

    return topicos, probabilidade


if __name__ == "__main__":
    dados = processamento()
    sentencas = sentencas(dados)
    modelo = modelo()
    topicos, probabilidade = treino(modelo, sentencas)