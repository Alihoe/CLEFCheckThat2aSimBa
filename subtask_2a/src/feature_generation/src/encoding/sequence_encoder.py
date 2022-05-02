import enum

import nltk
from nltk.tokenize import sent_tokenize

from src.feature_generation.src.encoding.infer_sent_encoder import InferSentEncoder
from src.feature_generation.src.encoding.sbert_encoder import SbertEncoder
from src.feature_generation.src.encoding.sim_cse_encoder import SimCSEEncoder
from src.feature_generation.src.encoding.universal_sentence_encoder import UniversalSentenceEncoder


class EncoderType(enum.Enum):
    sbert_encoder = 1
    infer_sent_encoder = 2
    universal_sentence_encoder = 3
    sim_cse_encoder = 4
    trans_encoder = 5


class TextEncodingMethod(enum.Enum):
    full_text = 1
    single_sentences = 2


class SequenceEncoder:

    def __init__(self, encoder_type, encoder_type_model):
        nltk.download('punkt')
        self.encoder_type = encoder_type
        if self.encoder_type == EncoderType.sbert_encoder.name:
            self.encoder = SbertEncoder(encoder_type_model)
        elif self.encoder_type == EncoderType.infer_sent_encoder.name:
            self.encoder = InferSentEncoder(encoder_type_model)
        elif self.encoder_type == EncoderType.universal_sentence_encoder.name:
            self.encoder = UniversalSentenceEncoder()
        elif self.encoder_type == EncoderType.sim_cse_encoder.name:
            self.encoder = SimCSEEncoder(encoder_type_model)

        else:
            raise ValueError('Choose between "sbert_encoder" with "small_model" or "large_model",'
                             ' "infer_sent_encoder" with "glove_embeddings" or "fast_text_embeddings"'
                             'or "universal_sentence_encoder" without a model specification ("") or "sim_cse_encoder" with "small_model" or "large_model".')

    def encode_sequence(self, sequence, encoding_method):
        if self.encoder_type == EncoderType.sbert_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return self.encoder.encode_sequence_model(sequence)
            if encoding_method == TextEncodingMethod.single_sentences.name:
                list_of_sentences = sent_tokenize(sequence)
                return list(self.encoder.encode_list_of_sequences_model(list_of_sentences))
        elif self.encoder_type == EncoderType.infer_sent_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return self.encoder.encode_sequence_model(sequence)
            if encoding_method == TextEncodingMethod.single_sentences.name:
                list_of_sentences = sent_tokenize(sequence)
                return list(self.encoder.encode_list_of_sequences_model(list_of_sentences))
        elif self.encoder_type == EncoderType.universal_sentence_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return self.encoder.encode_sequence_model(sequence)
            if encoding_method == TextEncodingMethod.single_sentences.name:
                list_of_sentences = sent_tokenize(sequence)
                return list(self.encoder.encode_list_of_sequences_model(list_of_sentences).tolist())
        elif self.encoder_type == EncoderType.sim_cse_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return self.encoder.encode_sequence_model(sequence)
            if encoding_method == TextEncodingMethod.single_sentences.name:
                list_of_sentences = sent_tokenize(sequence)
                return list(self.encoder.encode_list_of_sequences_model(list_of_sentences))
        elif self.encoder_type == EncoderType.trans_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return self.encoder.encode_sequence_model(sequence)
            if encoding_method == TextEncodingMethod.single_sentences.name:
                list_of_sentences = sent_tokenize(sequence)
                return list(self.encoder.encode_list_of_sequences_model(list_of_sentences))
        else:
            raise RuntimeError('Sequence Encoder not properly initialized.')

    def encode_list_of_sequences(self, list_of_sequences, encoding_method):
        if self.encoder_type == EncoderType.sbert_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return list(self.encoder.encode_list_of_sequences_model(list_of_sequences))
            if encoding_method == TextEncodingMethod.single_sentences.name:
                output = []
                for sequence in list_of_sequences:
                    list_of_sentences = sent_tokenize(sequence)
                    output.append(list(self.encoder.encode_list_of_sequences_model(list_of_sentences)))
                return output
        elif self.encoder_type == EncoderType.infer_sent_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return list(self.encoder.encode_list_of_sequences_model(list_of_sequences))
            if encoding_method == TextEncodingMethod.single_sentences.name:
                output = []
                for sequence in list_of_sequences:
                    list_of_sentences = sent_tokenize(sequence)
                    output.append(list(self.encoder.encode_list_of_sequences_model(list_of_sentences)))
                return output
        elif self.encoder_type == EncoderType.universal_sentence_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return list(self.encoder.encode_list_of_sequences_model(list_of_sequences))
            if encoding_method == TextEncodingMethod.single_sentences.name:
                output = []
                for sequence in list_of_sequences:
                    list_of_sentences = sent_tokenize(sequence)
                    output.append(list(self.encoder.encode_list_of_sequences_model(list_of_sentences)))
                return output
        elif self.encoder_type == EncoderType.sim_cse_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return list(self.encoder.encode_list_of_sequences_model(list_of_sequences))
            if encoding_method == TextEncodingMethod.single_sentences.name:
                output = []
                for sequence in list_of_sequences:
                    list_of_sentences = sent_tokenize(sequence)
                    output.append(list(self.encoder.encode_list_of_sequences_model(list_of_sentences)))
                return output
        elif self.encoder_type == EncoderType.trans_encoder.name:
            if encoding_method == TextEncodingMethod.full_text.name:
                return list(self.encoder.encode_list_of_sequences_model(list_of_sequences))
            if encoding_method == TextEncodingMethod.single_sentences.name:
                output = []
                for sequence in list_of_sequences:
                    list_of_sentences = sent_tokenize(sequence)
                    output.append(list(self.encoder.encode_list_of_sequences_model(list_of_sentences)))
                return output
        else:
            raise RuntimeError('Sequence Encoder not properly initialized.')

