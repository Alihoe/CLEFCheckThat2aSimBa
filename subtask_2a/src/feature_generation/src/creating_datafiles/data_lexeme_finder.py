import json
import os
import pandas as pd


class DataLexemeFinder:

    @staticmethod
    def find_all_lexemes(input_file, output_file, lexeme_finder):
        if os.path.isfile(input_file):
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            list_of_ids = []
            list_of_synonyms = []
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                list_of_ids.append(i_claim_id)
                list_of_synonyms.append(lexeme_finder.get_lexemes(i_claim))
            data_tuples = list(zip(list_of_ids, list_of_synonyms))
            df = pd.DataFrame(data_tuples, columns=['input_claim_ids', 'used_words_and_synonyms'])
            df.to_pickle(output_file)
        elif os.path.isdir(input_file):
            list_of_ver_claim_ids = []
            list_of_synonyms = []
            for json_file in os.listdir(input_file):
                json_file_path = input_file + '/' + json_file
                with open(json_file_path, 'r') as j:
                    v_claim = json.loads(j.read())
                list_of_ver_claim_ids.append(v_claim['vclaim_id'])
                list_of_synonyms.append(lexeme_finder.get_lexemes(v_claim['vclaim']))
            data_tuples = list(zip(list_of_ver_claim_ids, list_of_synonyms))
            df = pd.DataFrame(data_tuples, columns=['ver_claim_ids', 'used_words_and_synonyms'])
            df.to_pickle(output_file)
        else:
            raise ValueError('Input should be a file of tweets or a directory of v-claims.')
