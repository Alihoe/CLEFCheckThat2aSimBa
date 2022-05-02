import json
import os
import pandas as pd

from src.feature_generation.src.entity_fishing.entity_fisher import EntityFisher


class DataNamedEntityDisambiguator:

    @staticmethod
    def disambiguate_named_entities(input_file, output_file):
        entity_fisher = EntityFisher()
        if os.path.isfile(input_file):
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            list_of_ids = []
            list_of_named_entity_lists = []
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                list_of_ids.append(i_claim_id)
                try:
                    list_of_named_entity_lists.append(entity_fisher.get_named_entities_of_sentence(i_claim))
                except:
                    list_of_named_entity_lists.append([])
            data_tuples = list(zip(list_of_ids, list_of_named_entity_lists))
            df = pd.DataFrame(data_tuples, columns=['input_claim_ids', 'named_entities'])
            df.to_pickle(output_file)
        elif os.path.isdir(input_file):
            list_of_ver_claim_ids = []
            list_of_named_entity_lists = []
            for json_file in os.listdir(input_file):
                json_file_path = input_file + '/' + json_file
                with open(json_file_path, 'r') as j:
                    v_claim = json.loads(j.read())
                list_of_ver_claim_ids.append(v_claim['vclaim_id'])
                list_of_named_entity_lists.append(entity_fisher.get_named_entities_of_sentence(v_claim['vclaim']))
            data_tuples = list(zip(list_of_ver_claim_ids, list_of_named_entity_lists))
            df = pd.DataFrame(data_tuples, columns=['ver_claim_ids', 'named_entities'])
            df.to_pickle(output_file)
        else:
            raise ValueError('Input should be a file of tweets or a directory of v-claims.')
