

from . import ROOT_DIR
import torch
import enum
import nltk

from .encoding_data.infer_sent.models import InferSent


class InferSentWordEmbeddingType(enum.Enum):
    glove_embeddings = 1
    fast_text_embeddings = 2


class InferSentEncoder:

    def __init__(self, word_embedding_type, sentences_for_vocab):
        nltk.download('punkt')
        if word_embedding_type == InferSentWordEmbeddingType.glove_embeddings.name:
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
            self.model = InferSent(params_model)
            self.model.load_state_dict(torch.load(ROOT_DIR+'/encoding_data/infer_sent/infersent1.pkl'))
            self.model.set_w2v_path(ROOT_DIR+'/encoding_data/glove.840B.300d.txt')
            self.model.build_vocab(sentences_for_vocab, tokenize=True)
        elif word_embedding_type == InferSentWordEmbeddingType.fast_text_embeddings.name:
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
            self.model = InferSent(params_model)
            self.model.load_state_dict(torch.load(ROOT_DIR+'/encoding_data/infer_sent/infersent2.pkl'))
            self.model.set_w2v_path(ROOT_DIR+'/encoding_data/crawl-300d-2M.vec')
            self.model.build_vocab(sentences_for_vocab, tokenize=True)
        else:
            raise ValueError('Choose between "glove_embeddings" or "fast_text_embeddings".')

    def encode_sequence_model(self, sequence):
        return self.model.encode([sequence], tokenize=True)[0]

    def encode_list_of_sequences_model(self, list_of_sequences):
        return self.model.encode(list_of_sequences, tokenize=True)



