import pickle
from src.config import Filenames
from src.etl.preprocess import tokenize


def tonality(phrase: str) -> str:
    with open(Filenames.MODEL, 'rb') as fm:
        model = pickle.load(fm)

    with open(Filenames.VECTORIZER, 'rb') as fv:
        vectorizer = pickle.load(fv)

    return model.predict(vectorizer.transform([' '.join(tokenize(phrase))]).toarray())
