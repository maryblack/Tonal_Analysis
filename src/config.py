class Filenames:
    TRAIN = "./data/train/news_eval_train.xml"
    TEST = "./data/test/news_eval_test.xml"
    RESULTS = "./results"
    MODEL = "./src/model/tonality.dat"
    VECTORIZER = "./src/model/vectorizer.pk"


class Parameters:
    WORDS_TO_REMOVE = [
        'все',
        'нет',
        'ни',
        'ничего',
        'без',
        'никогда',
        'наконец',
        'больше',
        'хорошо',
        'лучше',
        'нельзя',
        'более',
        'всегда',
        'конечно',
        'всю',
        'такой',
        'впрочем',
        'так',
        'вот',
        'можно',
        'даже',
        'разве'
    ]
