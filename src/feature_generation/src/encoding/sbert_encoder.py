import enum
from sentence_transformers import SentenceTransformer


class SbertEncoderModel(enum.Enum):
    small_model = 1
    large_model = 2


class SbertEncoder:

    def __init__(self, model):

        if model == SbertEncoderModel.small_model.name:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif model == SbertEncoderModel.large_model.name:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        else:
            raise ValueError('Choose between "small_model" or "large_model".')

    def encode_sequence_model(self, sequence):
        return self.model.encode(sequence).flatten()

    def encode_list_of_sequences_model(self, sequence):
        return self.model.encode(sequence)
