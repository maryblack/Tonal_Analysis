import pickle
from src.config import Filenames
from src.etl.preprocess import tokenize


def tonality(phrase: str) -> str:
    with open("./model/tonality.dat", 'rb') as fm:
        model = pickle.load(fm)

    with open("./model/vectorizer.pk", 'rb') as fv:
        vectorizer = pickle.load(fv)

    return model.predict(vectorizer.transform([' '.join(tokenize(phrase))]).toarray())
