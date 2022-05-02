import enum
from simcse import SimCSE


class SimCSEEncoderModel(enum.Enum):
    small_model = 1
    large_model = 2


class SimCSEEncoder:

    def __init__(self, model):

        if model == SimCSEEncoderModel.small_model.name:
            self.model = SimCSE('princeton-nlp/unsup-simcse-bert-base-uncased')
        elif model == SimCSEEncoderModel.large_model.name:
            self.model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
        else:
            raise ValueError('Choose between "small_model" or "large_model".')

    def encode_sequence_model(self, sequence):
        return self.model.encode(sequence).numpy()

    def encode_list_of_sequences_model(self, sequence):
        return self.model.encode(sequence).numpy()



