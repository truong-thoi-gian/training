from enum import Enum
class ModelType(Enum):
    # word embedding model
    GLOVE_300 = "glove-wiki-gigaword-300.bin"
    GLOVE_50 = "glove-wiki-gigaword-50.bin"
    FASTTEXT_300 = "fasttext.bin"
    FASTTEXT_300_SMALL = "wiki-news-300d-1M-subword.bin"
    WORD2VEC_300 = "word2-vec-GoogleNews-vectors-negative300.bin"
    CRAWL = "crawl-300d-2M.vec"


