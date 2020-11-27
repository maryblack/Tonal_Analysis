import logging
import yaml
import io
from src.etl.preprocess import parse_xml, vectorized_corpus
from src.config import Filenames

logging.basicConfig(level=logging.DEBUG)


def main():
    logging.info("Running pipeline")
    citations_train = parse_xml(Filenames.TRAIN)
    citations_test = parse_xml(Filenames.TEST)


if __name__ == '__main__':
    main()
