import json
import os
import pandas as pd

from src.feature_generation.src.encoding.sequence_encoder import TextEncodingMethod, SequenceEncoder


class DataEncoder:

    def __init__(self, encoder_type, encoder_type_model, encoding_method, sentences_for_vocab=0):
        self.encoder = SequenceEncoder(encoder_type, encoder_type_model, sentences_for_vocab)
        if encoding_method is TextEncodingMethod.full_text.name or encoding_method is TextEncodingMethod.single_sentences.name:
            self.encoding_method = encoding_method
        else:
            raise ValueError('Choose encoding method "full_text" or "single_sentences".')

    def encode(self, input_file, output_file):
        if os.path.isfile(input_file):
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            list_of_ids = []
            list_of_texts = []
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                list_of_ids.append(i_claim_id)
                list_of_texts.append(i_claim)
            list_of_encodings = self.encoder.encode_list_of_sequences(list_of_texts, self.encoding_method)
            data_tuples = list(zip(list_of_ids, list_of_encodings))
            df = pd.DataFrame(data_tuples, columns=['tweet_ids', 'tweet_encodings'])
            df.to_pickle(output_file)
        elif os.path.isdir(input_file):
            list_of_ver_claim_ids = []
            list_of_ver_claim_texts = []
            for json_file in os.listdir(input_file):
                json_file_path = input_file + '/' + json_file
                with open(json_file_path, 'r') as j:
                    v_claim = json.loads(j.read())
                list_of_ver_claim_ids.append(v_claim['vclaim_id'])
                list_of_ver_claim_texts.append(v_claim['vclaim'])
            list_of_encodings = self.encoder.encode_list_of_sequences(list_of_ver_claim_texts, self.encoding_method)
            data_tuples = list(zip(list_of_ver_claim_ids, list_of_encodings))
            df = pd.DataFrame(data_tuples, columns=['ver_claim_ids', 'ver_claim_encodings'])
            df.to_pickle(output_file)
        else:
            raise ValueError('Input should be a file of tweets or a directory of v-claims.')

    @staticmethod
    def get_sentences_to_encode(input_file):
        if os.path.isfile(input_file):
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            list_of_ids = []
            list_of_texts = []
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                list_of_ids.append(i_claim_id)
                list_of_texts.append(i_claim)
            return list_of_texts
        elif os.path.isdir(input_file):
            list_of_ver_claim_ids = []
            list_of_ver_claim_texts = []
            for json_file in os.listdir(input_file):
                json_file_path = input_file + '/' + json_file
                with open(json_file_path, 'r') as j:
                    v_claim = json.loads(j.read())
                list_of_ver_claim_ids.append(v_claim['vclaim_id'])
                list_of_ver_claim_texts.append(v_claim['vclaim'])
            return list_of_ver_claim_texts
        else:
            raise ValueError('Input should be a file of tweets or a directory of v-claims.')
