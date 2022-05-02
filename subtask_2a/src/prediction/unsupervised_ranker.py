import pandas as pd
import numpy as np


class UnsupervisedRanker:

    def __init__(self, list_of_features):
        self.features = list_of_features

    def create_sim_score(self, feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        data_set.columns = ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse',
                            'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects']
        data_set['sequence_matcher'] = data_set['sequence_matcher'].astype('float')
        data_set['levenshtein'] = data_set['levenshtein'].astype('float')
        data_set['jacc_char'] = data_set['jacc_char'].astype('float')
        data_set['jacc_tok'] = data_set['jacc_tok'].astype('float')
        data_set['ne'] = data_set['ne'].astype('float')
        data_set['main_syns'] = data_set['main_syns'].astype('float')
        data_set['words'] = data_set['words'].astype('float')
        data_set['subjects'] = data_set['subjects'].astype('float')
        output_df = pd.DataFrame()
        for feature in self.features:
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
            else:
                current_sim_score = output_df
            current_feature = data_set[feature]
            if feature in ['sbert', 'infersent', 'universal', 'sim_cse']:
                if current_sim_score.empty:
                    sim_score = current_feature
                else:
                    sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            elif feature in ['sequence_matcher', 'jacc_char', 'jacc_tok', 'subjects']:
                if current_sim_score.empty:
                    sim_score = current_feature
                else:
                    current_feature = current_feature*100
                    sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            elif feature in ['levenshtein']:
                if current_sim_score.empty:
                    sim_score = current_feature
                else:
                    current_feature = current_feature/-100
                    sim_score = np.divide(current_sim_score, current_feature)
            elif feature in ['ne', 'main_syns', 'words']:
                if current_sim_score.empty:
                    sim_score = current_feature
                else:
                    sim_score = np.add(current_sim_score, current_feature)
            output_df['sim_score'] = sim_score
        output_df['i_claim_id'] = data_set['i_claim_id']
        output_df['ver_claim_id'] = data_set['ver_claim_id']
        output_df['QO'] = 'QO'
        output_df['rank'] = '1'
        output_df['tag'] = 'SimBa'
        output_df = output_df[['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag']]
        return output_df

    @staticmethod
    def rank_by_top_n_similarity_score(output, underlying_data, output_data, n=5):
        df = pd.DataFrame(columns=['i_claim_id', 'QO', 'ver_claim_id', 'rank', 'sim_score', 'tag'])
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
        list_of_iclaim_ids = underlying_df['iclaim_id'].tolist()
        for i_claim_id in list_of_iclaim_ids:
            this_i_claim_df = output[output.i_claim_id == i_claim_id]
            this_i_claim_df = this_i_claim_df.sort_values('sim_score', ascending=False)
            this_i_claim_df = this_i_claim_df.head(n=n)
            df = pd.concat([df, this_i_claim_df])
        df.to_csv(output_data, index=False, header=False, sep ='\t')

    def create_top_n_output_file(self, feature_set, underlying_data, output_data, n=5):
        feature_set_with_sim_scores = self.create_sim_score(feature_set)
        UnsupervisedRanker.rank_by_top_n_similarity_score(feature_set_with_sim_scores, underlying_data, output_data, n)

