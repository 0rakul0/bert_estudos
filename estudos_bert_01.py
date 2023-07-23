"""

Esse modelo tem o objetivo de responder perguntas com base em um corpus de texto

"""
import transformers
from transformers import pipeline

qea = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

texto = "Carl Sagan foi um cientista norte-americano. Sagan é autor de mais de 600 publicações científicas e também de mais de vinte livros de ciência e ficção científica."
pergunta = "Carl Sagan é um cientista brasileiro?"

resposta = qea(question=pergunta, context=texto)
print("Pergunta: ", pergunta)
print("Resposta: ", resposta['answer'])
print("Score: ", resposta['score'])


