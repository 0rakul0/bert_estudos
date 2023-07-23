"""
rede neurais convolucionais para PLN ou NLP
esse arquivo faz a limpeza dos dados
"""
import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup as bs
import seaborn as sns
import spacy as sp
from spacy.lang.en import STOP_WORDS as sw
import string
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import keras.datasets as tfds
from sklearn.model_selection import train_test_split


class pln():
    ### carregamento dos dados
    def carregamento(self):
        path = r'./trainingandtestdata/'
        #"0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
        cols = ['sentiment', 'id', 'date','query', 'user', 'text']
        data = pd.read_csv(f'{path}train.csv', header=None, names=cols, engine='python', encoding='latin1')
        return data

    def analise(self, data):
        print(data.shape)
        print(data['sentiment'].unique())
        sns.countplot(x=data['sentiment'].unique())
        plt.title('Contagem de Valores Únicos na Coluna "sentiment"')
        plt.show()

    def processamento(self, data, comentario=None):
        """
        essa função tem como objetivo fazer uma limpeza
        :param data: dataset
        :param comentario: ativa os cometarios de saida
        :return:
        """
        data.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)
        X = data.iloc[:, 1].values # : <- todos os registros , 1 <- todas as colunas, values <- transforma em um array
        y = data.iloc[:, 0].values
        if comentario:
            print("amostragem\n",data.head())
            print("tamanho de X <- previsores", X.shape)
            print(X[1])
            print(y[1])
            print("tamanho de y <- respostas", X.shape)
        return X, y

    def model(self, X,y, tamanho, comentario=None):
        """
        como temos muitos dados, precisamos diminuir o tamanho, logo use o paramentro tamanho para reter o que precisa
        1-t = tamanho retido
        exemplo 1 - 0.2 = 0.8 ficará retido 80% do total
        :param X: previsores
        :param y: classe
        :param tamanho: tamanho do que será retido
        :param comentario: ativa os comentarios de saida
        :return:
        """
        X, _, y, _ = train_test_split(X, y, test_size=(1-tamanho), stratify=y)

        unico_y, qte_y = np.unique(y, return_counts=True)
        unico_X, qte_x = np.unique(X, return_counts=True)

        if comentario:
            print("tamanho total para treinamento", X.shape)
            print("tamanho total para teste", y.shape)

            print("unicos e quantidade de y", unico_y, qte_y)
            print("unicos e quantidade de x", unico_X, qte_x)
        return X,y

    def limpeza(self, texto):
        texto = bs(texto, 'lxml').get_text()
        texto = re.sub(r"@[A-Za-z0-9]+", ' ',texto) #tira os nomes de usuarios
        texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto) # tira as urls
        texto = re.sub(r"[^a-zA-Z.!?]", ' ', texto) # remove impurezas tais como numeros etc
        texto = re.sub(r" +", ' ', texto) # remove espaços em brancos muito longos
        texto = texto.lower() # deixa o texto em minusculo
        return texto

    def modelo_spacy(self, texto):
        nlp = sp.load('en_core_web_sm')
        doc = nlp(texto)
        palavras = []
        for token in doc:
            palavras.append(token.text)
        palavras = [palavra for palavra in palavras if palavra not in sw and palavra not in string.punctuation and len(list(palavra)) > 1 ] # remove caracteres unicos, pontos
        palavras = ' '.join([str(elemento) for elemento in palavras]) # volta como string
        return palavras

    def limpeza_geral(self, X, y):
        data_texto = [self.modelo_spacy(self.limpeza(texto)) for texto in X]
        data_labels = y
        data_labels[data_labels == 4] = 1 # transforma para 0 ou 1
        return data_texto, data_labels

    def run(self):
        data = self.carregamento()
        # self.analise(data)
        X, y = self.processamento(data, comentario=False)
        X, y = self.model(X,y, 0.15, comentario=False)
        texto, resposta = p.limpeza_geral(X, y)

if __name__ == "__main__":
    p = pln()
    p.run()



