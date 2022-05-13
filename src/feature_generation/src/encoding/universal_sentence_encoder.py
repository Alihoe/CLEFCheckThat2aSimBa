import tensorflow_hub as hub


class UniversalSentenceEncoder:

    def __init__(self):
        encoder_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.model = encoder_model

    def encode_sequence_model(self, sequence):
        sequence = [sequence]
        return self.model(sequence)[0].numpy()

    def encode_list_of_sequences_model(self, list_of_sequences):
        return self.model(list_of_sequences).numpy()
