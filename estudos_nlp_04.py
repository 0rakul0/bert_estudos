"""
rede neurais convolucionais para PLN ou NLP
esse arquivo desenvolve o modelo da rede neural
"""
import numpy as np
import math
import multiprocessing as mp
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
from keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import modelo_dcnn


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

    def separador(self, X,y, tamanho, comentario=None):
        """
        como temos muitos dados, precisamos diminuir o tamanho, logo use o paramentro tamanho para reter o que precisa
        1-t = tamanho retido
        exemplo 1 - 0.2 = 0.8 ficará retido 80% do total
        :param X: previsores
        :param y: classe
        :param tamanho: tamanho que será do total
        :param comentario: ativa os comentarios de saida
        :return:
        """
        X, _, y, _ = train_test_split(X, y, test_size=(1-tamanho), stratify=y)

        unico_y, qte_y = np.unique(y, return_counts=True)
        unico_X, qte_x = np.unique(X, return_counts=True)

        if comentario:
            print("tamanho total para treinamento", X.shape)
            print("tamanho total para teste", y.shape)
        return X,y

    def limpeza(self, texto):
        texto = bs(texto, 'lxml').get_text()
        texto = re.sub(r"@[A-Za-z0-9]+", ' ',texto) #tira os nomes de usuarios
        texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto) # tira as urls
        texto = re.sub(r"[^a-zA-Z.!?]", ' ', texto) # remove impurezas tais como numeros etc
        texto = re.sub(r"\b\.*\b", '', texto)
        texto = re.sub(r"\b!*\b", '', texto)
        texto = re.sub(r"\b_\b", '', texto) # tira o _ das palavras
        texto = re.sub(r" +", ' ', texto) # remove espaços em brancos muito longos
        texto = texto.lower() # deixa o texto em minusculo
        return texto

    def modelo_spacy(self, texto):
        """
        usado para limpar o texto
        :param texto: texto não formatado
        :return: texto formatado
        """
        nlp = sp.load('en_core_web_sm')
        doc = nlp(texto)
        palavras = []
        for token in doc:
            palavras.append(token.text)
        palavras = [palavra for palavra in palavras if palavra not in sw and palavra not in string.punctuation and len(list(palavra)) > 1 ] # remove caracteres unicos, pontos
        palavras = ' '.join([str(elemento) for elemento in palavras]) # volta como string
        return palavras

    def count_cpu(self):
        n_c = mp.cpu_count()
        n_c = int(n_c/2)
        return n_c

    def processa_texto(self,texto):
        return self.modelo_spacy(self.limpeza(texto))

    def limpeza_geral(self, X, y, processo=None):
        """
        o parametro processo habilita a opção de multprocessamento, use com cuidado!!!

        :param X: recebe uma lista de sentenças
        :param y: recebe as respostas
        :param processo: numero dos nucleos de processadores para processamento paralelo
        :return:
        """
        if processo:
            p = mp.Pool(processes=processo)
            data_texto = p.map(self.processa_texto, X)
            p.close()
        else:
            data_texto = [self.processa_texto(i) for i in X]

        data_labels = y
        data_labels[data_labels == 4] = 1 # transforma para 0 ou 1
        return data_texto, data_labels

    def tonkenerizador(self, texto, treino=None):
        """
        esse é o nosso encoder, o objetivo dele é trazer as frases separalas em uma lista
        e dá um indentificador para cada palavra
        :param texto: lista com o texto limpo
        :return:
        """
        encode_on = r"./vocabs/vocab_tokenizer"

        if encode_on and treino==False:
            tokenizer_load = self.carrega_vocab()
        else:
            # build
            tokenizer_build = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(texto, target_vocab_size=2**15)
            tokenizer_build.save_to_file("./vocabs/vocab_tokenizer")
            print('salvando o vocab')
            tokenizer_load = self.carrega_vocab()

        return tokenizer_load

    def carrega_vocab(self):
        tokenizer_load = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./vocab_tokenizer')
        print('vocab carregado')
        return tokenizer_load

    def base_teste(self, X, y, tokenize):
        data_input = [tokenize.encode(s) for s in X[0:100]]
        for _ in range(5):
            print("encoder", data_input[random.randint(0, len(data_input)-1)])

    def padding(self, X, tokenize):
        """
        essa função serve para vermos quanto 0 deveremos preencher para cada linha
        :param X: recebe a base
        :param tokenize: recebe o encoder
        :return:
        """
        data_input = [tokenize.encode(s) for s in X]
        max_len = max([len(m) for m in data_input ])
        print(max_len)

        data_input = tf.keras.preprocessing.sequence.pad_sequences(data_input, value=0,
                                                                   padding='post', maxlen=max_len)
        return data_input

    def treinamento(self, data_input, resposta, tokenize, tratado=None):

        if tratado == None:
            data_labels = resposta
            data_labels[data_labels == 4] = 1  # transforma para 0 ou 1
            resposta = data_labels

        train_input, teste_input, train_label, teste_label = train_test_split(data_input, resposta,test_size=0.3, stratify=resposta)

        print("###### treino #####")
        print(train_input.shape)
        print(train_label.shape)

        print("###### teste #####")
        print(teste_input.shape)
        print(teste_label.shape)

        vocab = tokenize

        emb_dim = 200
        nb_filters = 100
        ffn_units = 256
        nb_classes = len(set(resposta))
        dropout = 0.2
        nb_epochs = 100
        bach_size = 128

        Dcnn = modelo_dcnn.DCNN(vocab_size=vocab.vocab_size, emb_dim=emb_dim,
                                nb_filters=nb_filters, ffn_units=ffn_units,
                                nb_classes=nb_classes, dropout_rate=dropout)
        if nb_classes == 2:
            Dcnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            Dcnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        save_model = './'
        ckpt = tf.train.Checkpoint(Dcnn=Dcnn)
        ckpt_manager = tf.train.CheckpointManager(ckpt, save_model, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('modelo restalrado')
        """
        com o modelo instanciado passamos os dados de treino, dados de resposta,
        o tamanho do Tamanho do lote batch_size (n de interações por lote),
        o numero de epocas e o validation_split= é para testar logo que possivel
        
        o mais importante é o val_accuracy <- ele quem define o real aprendizado,
        nosso modelo está apresentando uma taxa em média de 80% de acerto,
        
        o callback serve para não precisar rodar tudo e parar antes 
        """
        modelo_estrutura = './classificador_dcnn.json'
        modelo_pesos = './classificador_dcnn'
        if modelo_estrutura and modelo_pesos:
            # para carregar
            arquivo = open('./classificador_dcnn.json', 'r')
            estrutura_rede = arquivo.read()
            arquivo.close()

            Dcnn = model_from_json(estrutura_rede)
            history = Dcnn.load_weights('./classificador_dcnn')
        else:
            history = Dcnn.fit(train_input, train_label, batch_size=bach_size, epochs=nb_epochs, verbose=1,
                           validation_split=0.1)

            Dcnn.save('./modelo_Dcnn.keras')

            # essa função serve para salvar o modelo em um checkpoint mas o certo é salvar o modelo e os pesos
            ckpt_manager.save()

            classificador_dcnn = Dcnn.to_json()
            with open('classificador_dcnn.json', 'w') as json_file:
                json_file.write(classificador_dcnn)
            Dcnn.save_weights('classificador_dcnn', save_format='tf')

        resultados = Dcnn.evaluate(teste_input, teste_label, batch_size=bach_size)
        print("evaluate do Dcnn",resultados)

        y_pred_teste = Dcnn.predict(teste_input)
        y_pred_teste = (y_pred_teste > 0.5)
        print("teste de sentimento: ",y_pred_teste)

        cm = confusion_matrix(teste_label, y_pred_teste)
        print(cm)

        # Salvar o heatmap em um arquivo separado
        sns.heatmap(cm, annot=True)
        plt.title('Matriz de Confusão')
        plt.savefig('heatmap.png')  # Nome do arquivo que será salvo
        plt.show()

        # Salvar o gráfico de progresso de validação
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Progresso de Validação')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training loss', 'Validation loss'])
        plt.savefig('loss_progress.png')  # Nome do arquivo que será salvo
        plt.show()

        # Salvar o gráfico de progresso de validação da acurácia
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Progresso de Validação da Acurácia')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Training accuracy', 'Validation accuracy'])
        plt.savefig('accuracy_progress.png')  # Nome do arquivo que será salvo
        plt.show()

        teste = ['i love you', 'i hate you']
        for t in teste:
            texto = tokenize.encode(t)
            predictions = Dcnn([texto], training=False).numpy()
            print(predictions)


    def run(self):
        tokenize = self.carrega_vocab()
        data = self.carregamento()
        X, y = self.processamento(data, comentario=False)
        n_core = self.count_cpu()
        # self.analise(data)
        texto, resposta = self.separador(X,y, 0.9, comentario=True)
        # texto, resposta = p.limpeza_geral(texto, resposta, processo=n_core)
        if tokenize == None:
            tokenize = self.tonkenerizador(texto, treino=False)
            # self.base_teste(X,y,tokenize)
        data_input = self.padding(texto,tokenize)
        self.treinamento(data_input, resposta, tokenize)


if __name__ == "__main__":
    p = pln()
    p.run()




