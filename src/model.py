from enum import Enum

import gensim
from model_type import ModelType
import Setting

class Model:
    def __init__(self, model_type: ModelType):
        filePath = Setting.MODEL_LOCATION + "/" + model_type.value
        if model_type in [ModelType.FASTTEXT_300, ModelType.FASTTEXT_300_SMALL]:
            self.__model = gensim.models.fasttext.load_facebook_vectors(filePath)
        else:
            self.__model = gensim.models.KeyedVectors.load_word2vec_format(filePath, binary=True)
        self.__vocab = self.__model

    @property
    def model(self):
        return self.__model

    @property
    def vocab(self):
        return self.__vocab

    def get_dimension(self):
        return self.__model.vector_size

    def get_vector(self, word):
        return self.__model[word]
