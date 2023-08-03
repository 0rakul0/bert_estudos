import re
from bs4 import BeautifulSoup as bs

def limpeza_corpus(texto):
    texto = bs(texto, 'lxml').get_text()
    texto = re.sub(r"@[A-Za-z0-9]+", ' ', texto)  # tira os nomes de usuarios
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)  # tira as urls
    texto = re.sub(r"[^a-zA-Z.!?]", ' ', texto)  # remove impurezas tais como numeros etc
    texto = re.sub(r"\b\.*\b", '', texto)
    texto = re.sub(r"\b!*\b", '', texto)
    texto = re.sub(r"\b_\b", '', texto)  # tira o _ das palavras
    texto = re.sub(r" +", ' ', texto)  # remove espaços em brancos muito longos
    texto = texto.lower()  # deixa o texto em minusculo
    return texto

def limpeza(texto):
    texto = bs(texto, 'lxml').get_text()
    texto = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", texto)
    texto = re.sub(r".\$\$\$", '', texto)
    texto = re.sub(r" +",' ', texto)
    texto = texto.split('\n')
    return texto
