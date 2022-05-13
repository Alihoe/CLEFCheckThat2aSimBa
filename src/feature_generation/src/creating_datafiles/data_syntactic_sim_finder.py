import enum
import json
import os
import pandas as pd

from src.feature_generation.src.syntactical_evaluation.part_of_speech_evaluater import PartOfSpeechEvaluator


class DataSyntacticInfoFinderType(enum.Enum):
    pos_order = 1
    subjects = 2


class DataSyntacticInfoFinder:

    def __init__(self, type):
        if type == DataSyntacticInfoFinderType.pos_order.name:
            self.type = DataSyntacticInfoFinderType.pos_order.name
        elif type == DataSyntacticInfoFinderType.subjects.name:
            self.type = DataSyntacticInfoFinderType.subjects.name
        else:
            raise ValueError('Choose between "pos_order" or "subjects".')


    def find_all_syntact_info(self, input_file, output_file):
        if self.type == DataSyntacticInfoFinderType.subjects.name:
            if os.path.isfile(input_file):
                i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
                list_of_ids = []
                list_of_syntactic_elements = []
                for row in i_claim_data.iterrows():
                    i_claim_id = row[1][0]
                    i_claim = row[1][1]
                    list_of_ids.append(i_claim_id)
                    list_of_syntactic_elements.append(PartOfSpeechEvaluator.get_subject_of_sentence(i_claim))
                data_tuples = list(zip(list_of_ids, list_of_syntactic_elements))
                df = pd.DataFrame(data_tuples, columns=['input_claim_ids', 'pos'])
                df.to_pickle(output_file)
            elif os.path.isdir(input_file):
                list_of_ver_claim_ids = []
                list_of_syntactic_elements = []
                for json_file in os.listdir(input_file):
                    json_file_path = input_file + '/' + json_file
                    with open(json_file_path, 'r') as j:
                        v_claim = json.loads(j.read())
                    list_of_ver_claim_ids.append(v_claim['vclaim_id'])
                    print(v_claim['vclaim_id'])
                    list_of_syntactic_elements.append(PartOfSpeechEvaluator.get_subject_of_sentence(v_claim['vclaim']))
                data_tuples = list(zip(list_of_ver_claim_ids,list_of_syntactic_elements))
                df = pd.DataFrame(data_tuples, columns=['ver_claim_ids', 'pos'])
                df.to_pickle(output_file)
            else:
                raise ValueError('Input should be a file of tweets or a directory of v-claims.')