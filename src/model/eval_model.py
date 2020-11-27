import pickle
from src.config import Filenames
from src.etl.preprocess import tokenize


def tonality(phrase: str) -> str:
    model = pickle.load(open(Filenames.MODEL, 'rb'))
    vectorizer = pickle.load(open(Filenames.VECTORIZER, 'rb'))

    return model.predict(vectorizer.transform([' '.join(tokenize(phrase))]).toarray())
