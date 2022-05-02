import json
import os

import pandas as pd


class DataFormatter:

    @staticmethod
    def ver_claim_directory_to_dataframe(input_directory, output_df):
        list_of_ver_claim_ids = []
        list_of_ver_claim_texts = []
        for json_file in os.listdir(input_directory):
            json_file_path = input_directory + '/' + json_file
            with open(json_file_path, 'r') as j:
                v_claim = json.loads(j.read())
            list_of_ver_claim_ids.append(v_claim['vclaim_id'])
            list_of_ver_claim_texts.append(v_claim['vclaim'])
        data_tuples = list(zip(list_of_ver_claim_ids, list_of_ver_claim_texts))
        df = pd.DataFrame(data_tuples, columns=['ver_claim_ids', 'ver_claim_texts'])
        df.to_pickle(output_df)


    @staticmethod
    def create_top_ranked_output_from_top_1(top_1_input_dataframe, top_1_output_file):
        input_df = pd.read_pickle(top_1_input_dataframe)
        n_rows = len(input_df.index)
        list_of_nulls = ['0'] * n_rows
        list_of_ones = ['1'] * n_rows
        input_df['0'] = list_of_nulls
        input_df['score'] = list_of_ones
        input_df = input_df[['new_claim_id', '0', 'ver_claim_id', 'score']]
        input_df.to_csv(top_1_output_file, sep="\t", index=False, header=False)

    @staticmethod
    def create_top_ranked_output_from_top_n(top_n_input_dataframe, top_n_output_file):
        input_df = pd.read_pickle(top_n_input_dataframe)
        input_df.loc[len(input_df.index)] = ['dummy_iclaim_id', 'dummy_vclaim_id', 0]
        rank = 0
        ranked_ver_claim_id = ''
        i_claim_id = input_df.iloc[0, 0]
        for row in input_df.iterrows():
            current_id = row[1][0]
            current_v_claim_id = row[1][1]
            current_rank = float(row[1][2])
            if current_id != i_claim_id:
                with open(top_n_output_file, 'a', encoding='utf8') as output_file:
                    joined_list = "\t".join([i_claim_id, '0', ranked_ver_claim_id, '1'])
                    print(joined_list, file=output_file)
                i_claim_id = current_id
                rank = current_rank
                ranked_ver_claim_id = current_v_claim_id
            else:
                if current_rank > rank:
                    rank = current_rank
                    ranked_ver_claim_id = current_v_claim_id


    @staticmethod
    def create_rankings_datafile(predicted_rankings_file, output):
        predicted_rankings = pd.read_pickle(predicted_rankings_file)
        only_positive_rankings = predicted_rankings.loc[predicted_rankings['rank'] == 1]
        only_positive_rankings = only_positive_rankings.iloc[:,[0,1,len(only_positive_rankings.columns)-1]]
        n_rows = len(only_positive_rankings.index)
        list_of_nulls = ['0'] * n_rows
        only_positive_rankings['0'] = list_of_nulls
        only_positive_rankings = only_positive_rankings.rename(columns={only_positive_rankings .columns[0]: 'i_claim_id'})
        only_positive_rankings = only_positive_rankings.rename(columns={only_positive_rankings.columns[1]: 'ver_claim_id'})
        only_positive_rankings = only_positive_rankings[['i_claim_id', '0', 'ver_claim_id', 'rank']]
        list_of_ranked_i_claims = only_positive_rankings['i_claim_id'].tolist()
        test_data_df = pd.read_csv('data/original_tweet_data/test_data/tweets-test.tsv', sep='\t',
                                   names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_test_i_claims = test_data_df.iloc[:, 0].tolist()
        for i_claim in list_of_test_i_claims:
            if i_claim not in list_of_ranked_i_claims:
                row = pd.Series([i_claim, '0', 'no match', 0], index=only_positive_rankings.columns)
                only_positive_rankings = only_positive_rankings.append(row, ignore_index=True)
        only_positive_rankings.to_csv(output, index=False,  header=False, sep='\t')
