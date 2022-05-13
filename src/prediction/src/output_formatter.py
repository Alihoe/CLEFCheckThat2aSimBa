import pandas as pd
import numpy as np

from src.feature_generation.src.creating_datafiles.feature_set_maker import FeatureSetMaker


class OutputFormatter:

    @staticmethod
    def drop_all_but_top_ver_claims(input, output):
        df = pd.read_csv(input, sep='\t', dtype=str, names=['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag'])
        df = df.sort_values('sim_score', ascending=False)
        df = df.drop_duplicates(['i_claim_id'], keep='first')
        df = df.sort_values('i_claim_id')
        df.to_csv(output, index=False, header=False, sep='\t')

    @staticmethod
    def find_i_claim_df(i_claim_id, dataframe):
        return dataframe[dataframe['id'].str.startswith(i_claim_id+'_')]

    @staticmethod
    def all_ver_claim_ids_in_combined_id(i_claim_df):
        list_of_ver_claim_ids = []
        for row in i_claim_df.iterrows():
            ver_claims = row[1][0].split('_')[-2:]
            for ver_claim in ver_claims:
                if ver_claim not in list_of_ver_claim_ids:
                    list_of_ver_claim_ids.append(ver_claim)
        return list_of_ver_claim_ids


    @classmethod
    def rank_by_highest_sentence_embeddings_score(self, output, underlying_data, output_data):
        col = output.columns
        ses1 = col[2]
        ses2 = col[3]
        ses3 = col[4]
        ses4 = col[5]
        df = pd.DataFrame(columns=['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag'])
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = output[output.i_claim_id == i_claim_id]
            s1 = this_i_claim_df[ses1]
            s2 = this_i_claim_df[ses2]
            s3 = this_i_claim_df[ses3]
            s4 = this_i_claim_df[ses4]
            this_i_claim_df['sim_score'] = np.mean(np.array([s1, s2, s3, s4]), axis=0)
            max_score_row = this_i_claim_df.iloc[this_i_claim_df['sim_score'].argmax()]
            ver_claim = max_score_row['ver_claim_id']
            sim_score = max_score_row['sim_score']
            s_row = pd.Series([i_claim_id, 'QO', ver_claim, '1', sim_score, 'SimBa'], index=df.columns)
            df = df.append(s_row, ignore_index=True)
        df.sort_values('i_claim_id')
        df.to_csv(output_data, index=False, header=False, sep ='\t')

    @classmethod
    def rank_by_top_n_sentence_embeddings_score(self, output, underlying_data, output_data, n=5):
        col = output.columns
        ses1 = col[2]
        ses2 = col[3]
        ses3 = col[4]
        ses4 = col[5]
        df = pd.DataFrame(columns=['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag'])
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = output[output.i_claim_id == i_claim_id]
            s1 = this_i_claim_df[ses1]
            s2 = this_i_claim_df[ses2]
            s3 = this_i_claim_df[ses3]
            s4 = this_i_claim_df[ses4]
            this_i_claim_df['sim_score'] = np.mean(np.array([s1, s2, s3, s4]), axis=0)
            this_i_claim_df = this_i_claim_df.sort_values('sim_score', ascending=False)
            this_i_claim_df = this_i_claim_df.head(n=n)
            this_i_claim_df['QO'] = 'QO'
            this_i_claim_df['rank'] = '1'
            this_i_claim_df['tag'] = 'SimBa'
            this_i_claim_df = this_i_claim_df[['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag']]
            df = pd.concat([df, this_i_claim_df])
        df.to_csv(output_data, index=False, header=False, sep ='\t')

    @staticmethod
    def retransform_dataset_for_triple_classification_to_ranked_feature_dataset(triple_feature_set, underlying_data, output_data):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        col = triple_feature_set.columns
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = OutputFormatter.find_i_claim_df(i_claim_id, triple_feature_set)
            list_of_ver_claims = this_i_claim_df.iloc[:, 0].tolist()
            list_of_ver_claims_1 = [i.split('_')[-2]for i in list_of_ver_claims]
            list_of_ver_claims_2 = [i.split('_')[-1]for i in list_of_ver_claims]
            list_of_scores = this_i_claim_df.iloc[:, len(col) - 1].tolist()
            list_of_ver_claim_ids = OutputFormatter.all_ver_claim_ids_in_combined_id(this_i_claim_df)
            print(list_of_ver_claim_ids)
            ver_claim_scoring_dic = dict.fromkeys(list_of_ver_claim_ids, 0)
            i = 0
            for ver_claim in list_of_ver_claims_1:
                ver_claim_scoring_dic[ver_claim] = ver_claim_scoring_dic[ver_claim] + list_of_scores[i]
                i = i + 1
            i = 0
            for ver_claim in list_of_ver_claims_2:
                ver_claim_scoring_dic[ver_claim] = ver_claim_scoring_dic[ver_claim] - list_of_scores[i]
                i = i + 1
            max_key = max(ver_claim_scoring_dic, key=ver_claim_scoring_dic.get)
            if ver_claim_scoring_dic[max_key] == 0:
                max_key = '0'
            max_value = str(ver_claim_scoring_dic[max_key])
            print(ver_claim_scoring_dic)
            with open(output_data, 'a') as output_file:
                joined_list = "\t".join([i_claim_id, 'Q0', max_key, '1', max_value, 'SimBa'])
                print(joined_list, file=output_file)

    @staticmethod
    def transform_scored_dataset_of_triple_classification(triple_feature_set, underlying_data, output_data):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        col = triple_feature_set.columns
        triple_feature_set_transformed = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = OutputFormatter.find_i_claim_df(i_claim_id, triple_feature_set)
            list_of_ver_claims = this_i_claim_df.iloc[:, 0].tolist()
            list_of_ver_claims_1 = [i.split('_')[-2]for i in list_of_ver_claims]
            list_of_ver_claims_2 = [i.split('_')[-1]for i in list_of_ver_claims]
            list_of_scores = this_i_claim_df.iloc[:, len(col) - 1].tolist()
            list_of_ver_claim_ids = OutputFormatter.all_ver_claim_ids_in_combined_id(this_i_claim_df)
            ver_claim_scoring_dic = dict.fromkeys(list_of_ver_claim_ids, 0)
            i = 0
            for ver_claim in list_of_ver_claims_1:
                ver_claim_scoring_dic[ver_claim] = ver_claim_scoring_dic[ver_claim] + list_of_scores[i]
                i = i + 1
            i = 0
            for ver_claim in list_of_ver_claims_2:
                ver_claim_scoring_dic[ver_claim] = ver_claim_scoring_dic[ver_claim] - list_of_scores[i]
                i = i + 1
            this_claim_df = pd.DataFrame.from_dict(ver_claim_scoring_dic, orient='index').reset_index()
            this_claim_df.rename(columns={'index': 'ver_claim_id', 0: 'sim_score'}, inplace=True)
            this_claim_df['i_claim_id'] = i_claim_id
            this_claim_df = this_claim_df[['i_claim_id', 'ver_claim_id', 'sim_score']]
            triple_feature_set_transformed = pd.concat([triple_feature_set_transformed, this_claim_df])
        triple_feature_set_transformed.to_pickle(output_data)

    @classmethod
    def format_double_output(self, output_second_classifier, underlying_data, output_data):
        df = output_second_classifier
        df = df.loc[df['rank'] == 1]
        df['QO'] = 'QO'
        df['rank'] = '1'
        df['tag'] = 'SimBa'
        df = df[['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag']]
        df.reset_index(drop=True, inplace=True)
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        already_classified = df['i_claim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            if i_claim_id not in already_classified:
                this_i_claim_df = output_second_classifier[output_second_classifier.i_claim_id == i_claim_id]
                this_i_claim_df['sim_score'] = output_second_classifier['sim_score']
                max_score_row = this_i_claim_df.iloc[this_i_claim_df['sim_score'].argmax()]
                ver_claim = max_score_row['ver_claim_id']
                sim_score = max_score_row['sim_score']
                s_row = pd.Series([i_claim_id, 'QO', ver_claim, '1', sim_score, 'SimBa'], index=df.columns)
                df = df.append(s_row, ignore_index=True)
        df.sort_values('i_claim_id')
        df['sim_score'] = df['sim_score'].div(10)
        df.to_csv(output_data, index=False, header=False, sep='\t')

    @classmethod
    def format_binary_output(self, output, underlying_data, output_data, n=5):
        df = output
        df = df.loc[df['rank'] == 1]
        s1 = df['sbert'].to_numpy()
        s2 = df['infersent'].to_numpy()
        s4 = df['sim_cse'].to_numpy()
        df['sim_score'] = np.mean(np.array([s1, s2, s4]), axis=0)
        df['QO'] = 'QO'
        df['rank'] = '1'
        df['tag'] = 'SimBa'
        df = df[['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag']]
        df.reset_index(drop=True, inplace=True)
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        already_classified = df['i_claim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            if i_claim_id not in already_classified:
                this_i_claim_df = output[output.i_claim_id == i_claim_id]
                s1 = this_i_claim_df['sbert'].to_numpy()
                s2 = this_i_claim_df['infersent'].to_numpy()
                s4 = this_i_claim_df['sim_cse'].to_numpy()
                this_i_claim_df['sim_score'] = np.mean(np.array([s1, s2, s4]), axis=0)
                this_i_claim_df = this_i_claim_df.sort_values('sim_score', ascending=False)
                this_i_claim_df = this_i_claim_df.head(n=n)
                this_i_claim_df['QO'] = 'QO'
                this_i_claim_df['rank'] = '1'
                this_i_claim_df['tag'] = 'SimBa'
                this_i_claim_df = this_i_claim_df[['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag']]
                df = pd.concat([df, this_i_claim_df])
        df = df.sort_values('i_claim_id')
        df.to_csv(output_data, index=False, header=False, sep='\t')

    @classmethod
    def format_probability_output(self, df, underlying_data, output_data):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = df[df.i_claim_id == i_claim_id]
            max_score_row = this_i_claim_df.iloc[this_i_claim_df['rank'].argmax()]
            ver_claim = max_score_row['ver_claim_id']
            sim_score = str(max_score_row['rank']*100)
            with open(output_data, 'a') as output_file:
                joined_list = "\t".join([i_claim_id, 'Q0', ver_claim, '1', sim_score, 'SimBa'])
                print(joined_list, file=output_file)

