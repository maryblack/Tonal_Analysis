import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import string
import pymorphy2
import re

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()


class Citation:
    def __init__(self, words, evaluation: str):
        self.words = words
        self.eval = evaluation

    def __str__(self):
        return f'{self.eval}: {self.words}'

    def __repr__(self):
        return self.__str__()


def tokenize(file_text, remove_words=False):
    tokens = nltk.word_tokenize(file_text)
    tokens = [i for i in tokens if (i not in string.punctuation)]
    stop_words = stopwords.words('russian')
    if remove_words:
        words_to_remove = ['все', 'нет', 'ни', 'ничего', 'без', 'никогда', 'наконец', 'больше', 'хорошо', 'лучше',
                           'нельзя', 'более', 'всегда', 'конечно', 'всю', 'такой', 'впрочем', 'так', 'вот', 'можно',
                           'даже', 'разве']
        for word in words_to_remove:
            stop_words.remove(word)
    tokens = [morph.parse(re.sub(r'[^\w\s]', '', i).lower())[0].normal_form for i in tokens if (i not in stop_words)]
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]
    for item in tokens:
        if '' == item or item.isspace():
            while item in tokens:
                tokens.remove(item)
    return tokens


def parse_xml(file: str) -> list:
    tree = ET.parse(file)
    root = tree.getroot()
    corpus = []
    citations = []
    for elem in root.iter('speech'):
        corpus.append(tokenize(elem.text))
    i = 0
    for elem in root.iter('evaluation'):
        pair_eval = elem.text.replace("\n", "")
        pair_eval = ''.join(pair_eval.split())
        # print (corpus[i])
        if pair_eval in ['0', '+', '-']:
            citation = Citation(corpus[i], pair_eval)
            citations.append(citation)

        i += 1
        # evaluate_id.append(elem.text)
    return citations


def vectorized_corpus(parsed_list_train, parsed_list_test):
    out_list_train = []
    out_list_test = []
    eval_list = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for cite in parsed_list_train:
        citation = cite.words
        evaluation = cite.eval
        citation = ' '.join(citation).strip()
        out_list_train.append(citation)
        y_train.append(evaluation)

    for cite in parsed_list_test:
        citation = cite.words
        evaluation = cite.eval
        citation = ' '.join(citation).strip()
        out_list_test.append(citation)
        y_test.append(evaluation)

    vectorizer = TfidfVectorizer()  # tf-idf
    train_data = vectorizer.fit_transform(out_list_train)
    X_train = train_data.toarray()

    test_data = vectorizer.transform(out_list_test)
    X_test = test_data.toarray()

    return X_train, y_train, X_test, y_test, vectorizer
