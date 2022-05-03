import enum
import pandas as pd
import numpy as np

from src.feature_generation import Features, sbert_sims_test_pp1, infersent_sims_test_pp1, universal_sims_test_pp1, \
    sim_cse_sims_test_pp1, seq_match_test_pp1, jacc_chars_test_pp1, jacc_tokens_test_pp1, subjects_test_pp1, \
    levenshtein_test_pp1, ne_ne_ratio_sims_test_pp1, ne_token_ratio_sims_test_pp1, main_syms_ratio_sims_test_pp1, \
    words_ratio_sims_test_pp1
from src.feature_generation.file_paths.TEST.TEST_file_names import sbert_sims_test_TEST, infersent_sims_test_TEST, \
    universal_sims_test_TEST, sim_cse_sims_test_TEST, seq_match_test_TEST, jacc_chars_test_TEST, jacc_tokens_test_TEST, \
    subjects_test_TEST, levenshtein_test_TEST, ne_ne_ratio_sims_test_TEST, ne_token_ratio_sims_test_TEST, \
    main_syms_ratio_sims_test_TEST, words_ratio_sims_test_TEST


class Data(enum.Enum):
    pp1 = 1
    TEST = 2


class UnsupervisedFeatureSetGenerator:

    def __init__(self, list_of_features, data_key_word):
        self.features = list_of_features
        if data_key_word == Data.pp1.name:
            self.data = Data.pp1.name
        elif data_key_word == Data.TEST.name:
            self.data = Data.TEST.name

    def create_sim_score(self):
        output_df = pd.DataFrame()
        if Features.sbert.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = sbert_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = sbert_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
                print(sim_score)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.infersent.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = infersent_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = infersent_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
                print(sim_score)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.universal.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = universal_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = universal_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.sim_cse.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = sim_cse_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = sim_cse_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.seq_match.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = seq_match_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = seq_match_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            current_feature = current_feature * 100
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.jacc_chars.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = jacc_chars_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = jacc_chars_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            current_feature = current_feature * 100
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.jacc_tokens.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = jacc_tokens_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = jacc_tokens_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            current_feature = current_feature * 100
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.subjects.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = subjects_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = subjects_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            current_feature = current_feature * 100
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.levenshtein.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = levenshtein_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = levenshtein_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            current_feature = current_feature * 100
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.ne_ne_ratio.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = ne_ne_ratio_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = ne_ne_ratio_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.ne_token_ratio.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = ne_token_ratio_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = ne_token_ratio_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.main_syms_ratio.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = main_syms_ratio_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = main_syms_ratio_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        if Features.words_ratio.name in self.features:
            if self.data == Data.pp1.name:
                sim_score_file = words_ratio_sims_test_pp1
            elif self.data == Data.TEST.name:
                sim_score_file = words_ratio_sims_test_TEST
            sim_score_df = pd.read_pickle(sim_score_file)
            current_feature = sim_score_df['sim_score']
            if not output_df.empty:
                current_sim_score = output_df['sim_score']
                sim_score = np.mean(np.array([current_sim_score, current_feature]), axis=0)
            else:
                sim_score = current_feature
            output_df['sim_score'] = sim_score
        output_df['i_claim_id'] = sim_score_df['i_claim_id']
        output_df['ver_claim_id'] = sim_score_df['ver_claim_id']
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
        df.to_csv(output_data, index=False, header=False, sep='\t')

    def create_top_n_output_file(self, underlying_data, output_data, n=5):
        feature_set_with_sim_scores = self.create_sim_score()
        UnsupervisedFeatureSetGenerator.rank_by_top_n_similarity_score(feature_set_with_sim_scores, underlying_data, output_data, n)
