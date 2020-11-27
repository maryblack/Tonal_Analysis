import logging
import pickle
from sklearn.linear_model import LogisticRegression
from src.etl.preprocess import parse_xml, vectorized_corpus
from src.config import Filenames

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)


def main():
    logging.info("Running pipeline")

    logging.info("ETL started")
    citations_train = parse_xml(Filenames.TRAIN)
    citations_test = parse_xml(Filenames.TEST)

    logging.info("Vectorize data")
    X_train, y_train, X_test, y_test, vectorizer = vectorized_corpus(citations_train, citations_test)

    logging.info("Train model")
    model = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True, tol=0.0001,
                               intercept_scaling=1, max_iter=100, penalty='l2', random_state=0, solver='saga',
                               multi_class='ovr', warm_start=True)

    model.fit(X_train, y_train)
    logging.info(model.score(X_test, y_test))

    logging.info("Save model and vectorizer")
    pickle.dump(model, open(Filenames.MODEL, 'wb'))
    pickle.dump(vectorizer, open(Filenames.VECTORIZER, 'wb'))

if __name__ == '__main__':
    main()
