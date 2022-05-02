import pandas as pd

from src.feature_generation.file_paths.TEST.TEST_file_names import sbert_encodings_training_TEST, \
    sbert_sims_training_TEST_tsv, sbert_encodings_test_TEST, sbert_sims_test_TEST, sbert_sims_test_TEST_tsv, \
    sbert_sims_training_TEST, infersent_encodings_training_TEST, infersent_sims_training_TEST, \
    infersent_sims_training_TEST_tsv, infersent_encodings_test_TEST, infersent_sims_test_TEST, \
    infersent_sims_test_TEST_tsv, universal_encodings_training_TEST, universal_sims_training_TEST, \
    universal_sims_training_TEST_tsv, universal_encodings_test_TEST, universal_sims_test_TEST, \
    universal_sims_test_TEST_tsv, sim_cse_encodings_training_TEST, sim_cse_sims_training_TEST, \
    sim_cse_sims_training_TEST_tsv, sim_cse_encodings_test_TEST, sim_cse_sims_test_TEST, sim_cse_sims_test_TEST_tsv, \
    seq_match_training_TEST, seq_match_training_TEST_tsv, seq_match_test_TEST, seq_match_test_TEST_tsv, \
    levenshtein_training_TEST, levenshtein_training_TEST_tsv, levenshtein_test_TEST, levenshtein_test_TEST_tsv, \
    jacc_chars_training_TEST, jacc_chars_training_TEST_tsv, jacc_chars_test_TEST, jacc_chars_test_TEST_tsv, \
    jacc_tokens_training_TEST, jacc_tokens_training_TEST_tsv, jacc_tokens_test_TEST, jacc_tokens_test_TEST_tsv, \
    ne_training_TEST, ne_sims_training_TEST, ne_sims_training_TEST_tsv, ne_test_TEST, ne_sims_test_TEST, \
    ne_sims_test_TEST_tsv, main_syms_training_TEST, main_syms_sims_training_TEST, main_syms_sims_training_TEST_tsv, \
    main_syms_test_TEST, main_syms_sims_test_TEST, main_syms_sims_test_TEST_tsv, words_training_TEST, \
    words_sims_training_TEST, words_sims_training_TEST_tsv, words_test_TEST, words_sims_test_TEST, \
    words_sims_test_TEST_tsv, subjects_training_TEST, subjects_sims_training_TEST, subjects_sims_training_TEST_tsv, \
    subjects_test_TEST, subjects_sims_test_TEST, subjects_sims_test_TEST_tsv, top_n_sbert_sims_training_TEST, \
    top_n_infersent_sims_training_TEST, top_n_universal_sims_training_TEST, top_n_sim_cse_sims_training_TEST, \
    top_n_infersent_sims_test_TEST, top_n_universal_sims_test_TEST, top_n_sbert_sims_test_TEST, \
    top_n_sim_cse_sims_test_TEST, top_50_sbert_sims_training_TEST_df, top_50_infersent_sims_training_TEST_df, \
    top_50_sim_cse_sims_training_TEST_df, top_50_sbert_sims_training_TEST_tsv, top_50_infersent_sims_training_TEST_tsv, \
    top_50_universal_sims_training_TEST_tsv, top_50_sim_cse_sims_training_TEST_tsv, \
    top_50_universal_sims_training_TEST_df, top_50_sbert_sims_test_TEST_df, top_50_sbert_sims_test_TEST_tsv, \
    top_50_infersent_sims_test_TEST_df, top_50_universal_sims_test_TEST_df, top_50_sim_cse_sims_test_TEST_df, \
    top_50_infersent_sims_test_TEST_tsv, top_50_universal_sims_test_TEST_tsv, top_50_sim_cse_sims_test_TEST_tsv, \
    ne_ne_ratio_sims_training_TEST, ne_ne_ratio_sims_training_TEST_tsv, ne_ne_ratio_sims_test_TEST, \
    ne_ne_ratio_sims_test_TEST_tsv, token_number_sims_training_TEST, token_number_sims_training_TEST_tsv, \
    token_number_sims_test_TEST, token_number_sims_test_TEST_tsv, token_number_training_TEST, token_number_test_TEST
from src.feature_generation.file_paths.pp2.pp2_files import sbert_encodings_training_pp2, sbert_sims_training_pp2, \
    sbert_sims_training_pp2_tsv, sbert_encodings_test_pp2, sbert_sims_test_pp2, sbert_sims_test_pp2_tsv, \
    infersent_encodings_training_pp2, infersent_sims_training_pp2, infersent_sims_training_pp2_tsv, \
    infersent_encodings_test_pp2, infersent_sims_test_pp2, infersent_sims_test_pp2_tsv, \
    universal_encodings_training_pp2, universal_sims_training_pp2, universal_sims_training_pp2_tsv, \
    universal_encodings_test_pp2, universal_sims_test_pp2, universal_sims_test_pp2_tsv, sim_cse_encodings_training_pp2, \
    sim_cse_sims_training_pp2, sim_cse_sims_training_pp2_tsv, sim_cse_encodings_test_pp2, sim_cse_sims_test_pp2, \
    sim_cse_sims_test_pp2_tsv, seq_match_training_pp2, seq_match_training_pp2_tsv, seq_match_test_pp2, \
    seq_match_test_pp2_tsv, levenshtein_training_pp2, levenshtein_training_pp2_tsv, levenshtein_test_pp2, \
    levenshtein_test_pp2_tsv, jacc_chars_training_pp2, jacc_chars_training_pp2_tsv, jacc_chars_test_pp2, \
    jacc_chars_test_pp2_tsv, jacc_tokens_training_pp2, jacc_tokens_training_pp2_tsv, jacc_tokens_test_pp2, \
    jacc_tokens_test_pp2_tsv, ne_training_pp2, ne_sims_training_pp2, ne_sims_training_pp2_tsv, ne_test_pp2, \
    ne_sims_test_pp2, ne_sims_test_pp2_tsv, main_syms_training_pp2, main_syms_sims_training_pp2, \
    main_syms_sims_training_pp2_tsv, main_syms_test_pp2, main_syms_sims_test_pp2, main_syms_sims_test_pp2_tsv, \
    words_training_pp2, words_sims_training_pp2, words_sims_training_pp2_tsv, words_test_pp2, words_sims_test_pp2, \
    words_sims_test_pp2_tsv, subjects_training_pp2, subjects_sims_training_pp2, subjects_sims_training_pp2_tsv, \
    subjects_test_pp2, subjects_sims_test_pp2, subjects_sims_test_pp2_tsv, top_n_sbert_sims_training_pp2, \
    top_n_infersent_sims_training_pp2, top_n_universal_sims_training_pp2, top_n_sim_cse_sims_training_pp2, \
    top_n_sbert_sims_test_pp2, top_n_infersent_sims_test_pp2, top_n_universal_sims_test_pp2, \
    top_n_sim_cse_sims_test_pp2, top_50_sbert_sims_training_pp2_df, top_50_infersent_sims_training_pp2_df, \
    top_50_universal_sims_training_pp2_df, top_50_sim_cse_sims_training_pp2_df, top_50_sbert_sims_test_pp2_df, \
    top_50_infersent_sims_test_pp2_df, top_50_universal_sims_test_pp2_df, top_50_sim_cse_sims_test_pp2_df, \
    top_50_sbert_sims_training_pp2_tsv, top_50_infersent_sims_training_pp2_tsv, top_50_universal_sims_training_pp2_tsv, \
    top_50_sim_cse_sims_training_pp2_tsv, top_50_sbert_sims_test_pp2_tsv, top_50_infersent_sims_test_pp2_tsv, \
    top_50_universal_sims_test_pp2_tsv, top_50_sim_cse_sims_test_pp2_tsv, ne_ne_ratio_sims_training_pp2, \
    ne_ne_ratio_sims_training_pp2_tsv, ne_ne_ratio_sims_test_pp2, ne_ne_ratio_sims_test_pp2_tsv, \
    token_number_sims_training_pp2, token_number_sims_training_pp2_tsv, token_number_sims_test_pp2, \
    token_number_sims_test_pp2_tsv, token_number_training_pp2, token_number_test_pp2
from src.feature_generation.src.creating_datafiles.data_sim_score_generator import SimilarityScoreDataGenerator
from src.feature_generation import Features, sbert_encodings_training_pp1, sbert_sims_training_pp1, \
    sbert_sims_training_pp1_tsv, sbert_encodings_test_pp1, sbert_sims_test_pp1, sbert_sims_test_pp1_tsv, \
    sbert_encodings_vclaims_pp1, infersent_encodings_vclaims_pp1, infersent_encodings_training_pp1, \
    infersent_sims_training_pp1, infersent_sims_training_pp1_tsv, infersent_encodings_test_pp1, infersent_sims_test_pp1, \
    infersent_sims_test_pp1_tsv, universal_encodings_vclaims_pp1, universal_encodings_training_pp1, \
    universal_sims_training_pp1, universal_sims_training_pp1_tsv, universal_encodings_test_pp1, universal_sims_test_pp1, \
    universal_sims_test_pp1_tsv, sim_cse_encodings_vclaims_pp1, sim_cse_encodings_training_pp1, \
    sim_cse_sims_training_pp1, sim_cse_sims_training_pp1_tsv, sim_cse_encodings_test_pp1, sim_cse_sims_test_pp1, \
    sim_cse_sims_test_pp1_tsv, seq_match_training_pp1, seq_match_training_pp1_tsv, seq_match_test_pp1, \
    seq_match_test_pp1_tsv, v_claims_df, levenshtein_training_pp1, levenshtein_training_pp1_tsv, levenshtein_test_pp1, \
    levenshtein_test_pp1_tsv, jacc_chars_training_pp1, jacc_chars_training_pp1_tsv, jacc_chars_test_pp1, \
    jacc_chars_test_pp1_tsv, jacc_tokens_training_pp1, jacc_tokens_training_pp1_tsv, jacc_tokens_test_pp1, \
    jacc_tokens_test_pp1_tsv, ne_training_pp1, ne_vclaims_pp1, ne_sims_test_pp1, ne_sims_training_pp1, \
    ne_sims_training_pp1_tsv, ne_test_pp1, ne_sims_test_pp1_tsv, main_syms_vclaims_pp1, main_syms_training_pp1, \
    main_syms_sims_training_pp1, main_syms_sims_training_pp1_tsv, main_syms_test_pp1, main_syms_sims_test_pp1, \
    main_syms_sims_test_pp1_tsv, words_vclaims_pp1, words_training_pp1, words_sims_training_pp1, \
    words_sims_training_pp1_tsv, words_test_pp1, words_sims_test_pp1, words_sims_test_pp1_tsv, subjects_vclaims_pp1, \
    subjects_training_pp1, subjects_sims_training_pp1, subjects_sims_training_pp1_tsv, subjects_test_pp1, \
    subjects_sims_test_pp1, subjects_sims_test_pp1_tsv, top_n_sbert_sims_training_pp1, \
    top_n_infersent_sims_training_pp1, top_n_universal_sims_training_pp1, top_n_sim_cse_sims_training_pp1, \
    top_50_sbert_sims_training_pp1_df, top_50_sbert_sims_training_pp1_tsv, top_50_infersent_sims_training_pp1_df, \
    top_50_infersent_sims_training_pp1_tsv, top_50_universal_sims_training_pp1_df, \
    top_50_universal_sims_training_pp1_tsv, top_50_sim_cse_sims_training_pp1_df, top_50_sim_cse_sims_training_pp1_tsv, \
    top_n_sbert_sims_test_pp1, top_n_infersent_sims_test_pp1, top_n_universal_sims_test_pp1, \
    top_n_sim_cse_sims_test_pp1, top_50_sbert_sims_test_pp1_df, top_50_sbert_sims_test_pp1_tsv, \
    top_50_infersent_sims_test_pp1_df, top_50_infersent_sims_test_pp1_tsv, top_50_universal_sims_test_pp1_df, \
    top_50_universal_sims_test_pp1_tsv, top_50_sim_cse_sims_test_pp1_df, top_50_sim_cse_sims_test_pp1_tsv, \
    ne_ne_ratio_sims_training_pp1, ne_ne_ratio_sims_training_pp1_tsv, ne_ne_ratio_sims_test_pp1, \
    ne_ne_ratio_sims_test_pp1_tsv, token_number_sims_training_pp1, token_number_sims_training_pp1_tsv, \
    token_number_sims_test_pp1, token_number_sims_test_pp1_tsv, token_number_vclaims_pp1, token_number_training_pp1, \
    token_number_test_pp1


class PairSimilarityFeatureGenerator:


    @staticmethod
    def compute_top_n_sentence_embeddings_features(data_set, n):
        if 'train' in data_set or 'dev' in data_set:
            sbert_all_sims = sbert_sims_training_pp1
            infersent_all_sims = infersent_sims_training_pp1
            universal_all_sims = universal_sims_training_pp1
            sim_cse_all_sims = sim_cse_sims_training_pp1
            top_n_sbert = top_n_sbert_sims_training_pp1
            top_n_infersent = top_n_infersent_sims_training_pp1
            top_n_universal = top_n_universal_sims_training_pp1
            top_n_sim_cse = top_n_sim_cse_sims_training_pp1
            if 'pp2' in data_set:
                sbert_all_sims = sbert_sims_training_pp2
                infersent_all_sims = infersent_sims_training_pp2
                universal_all_sims = universal_sims_training_pp2
                sim_cse_all_sims = sim_cse_sims_training_pp2
                top_n_sbert = top_n_sbert_sims_training_pp2
                top_n_infersent = top_n_infersent_sims_training_pp2
                top_n_universal = top_n_universal_sims_training_pp2
                top_n_sim_cse = top_n_sim_cse_sims_training_pp2
            elif 'TEST' in data_set:
                sbert_all_sims = sbert_sims_training_TEST
                infersent_all_sims = infersent_sims_training_TEST
                universal_all_sims = universal_sims_training_TEST
                sim_cse_all_sims = sim_cse_sims_training_TEST
                top_n_sbert = top_n_sbert_sims_training_TEST
                top_n_infersent = top_n_infersent_sims_training_TEST
                top_n_universal = top_n_universal_sims_training_TEST
                top_n_sim_cse = top_n_sim_cse_sims_training_TEST
        elif 'test' in data_set:
            sbert_all_sims = sbert_sims_test_pp1
            infersent_all_sims = infersent_sims_test_pp1
            universal_all_sims = universal_sims_test_pp1
            sim_cse_all_sims = sim_cse_sims_test_pp1
            top_n_sbert = top_n_sbert_sims_test_pp1
            top_n_infersent = top_n_infersent_sims_test_pp1
            top_n_universal = top_n_universal_sims_test_pp1
            top_n_sim_cse = top_n_sim_cse_sims_test_pp1
            if 'pp2' in data_set:
                sbert_all_sims = sbert_sims_test_pp2
                infersent_all_sims = infersent_sims_test_pp2
                universal_all_sims = universal_sims_test_pp2
                sim_cse_all_sims = sim_cse_sims_test_pp2
                top_n_sbert = top_n_sbert_sims_test_pp2
                top_n_infersent = top_n_infersent_sims_test_pp2
                top_n_universal = top_n_universal_sims_test_pp2
                top_n_sim_cse = top_n_sim_cse_sims_test_pp2
            elif 'TEST' in data_set:
                sbert_all_sims = sbert_sims_test_TEST
                infersent_all_sims = infersent_sims_test_TEST
                universal_all_sims = universal_sims_test_TEST
                sim_cse_all_sims = sim_cse_sims_test_TEST
                top_n_sbert = top_n_sbert_sims_test_TEST
                top_n_infersent = top_n_infersent_sims_test_TEST
                top_n_universal = top_n_universal_sims_test_TEST
                top_n_sim_cse = top_n_sim_cse_sims_test_TEST
        SimilarityScoreDataGenerator.generate_top_n_from_top_all_sim_scores(sbert_all_sims, n,
                                                                            top_n_sbert)
        SimilarityScoreDataGenerator.generate_top_n_from_top_all_sim_scores(infersent_all_sims, n,
                                                                            top_n_infersent)
        SimilarityScoreDataGenerator.generate_top_n_from_top_all_sim_scores(universal_all_sims, n,
                                                                            top_n_universal)
        SimilarityScoreDataGenerator.generate_top_n_from_top_all_sim_scores(sim_cse_all_sims, n,
                                                                            top_n_sim_cse)
        if 'pp2' in data_set:
            if n == 50 and ('train' in data_set or 'dev' in data_set):
                pd.read_pickle(top_50_sbert_sims_training_pp2_df).to_csv(top_50_sbert_sims_training_pp2_tsv)
                pd.read_pickle(top_50_infersent_sims_training_pp2_df).to_csv(top_50_infersent_sims_training_pp2_tsv)
                pd.read_pickle(top_50_universal_sims_training_pp2_df).to_csv(top_50_universal_sims_training_pp2_tsv)
                pd.read_pickle(top_50_sim_cse_sims_training_pp2_df).to_csv(top_50_sim_cse_sims_training_pp2_tsv)
            if n == 50 and 'test' in data_set:
                pd.read_pickle(top_50_sbert_sims_test_pp2_df).to_csv(top_50_sbert_sims_test_pp2_tsv)
                pd.read_pickle(top_50_infersent_sims_test_pp2_df).to_csv(top_50_infersent_sims_test_pp2_tsv)
                pd.read_pickle(top_50_universal_sims_test_pp2_df).to_csv(top_50_universal_sims_test_pp2_tsv)
                pd.read_pickle(top_50_sim_cse_sims_test_pp2_df).to_csv(top_50_sim_cse_sims_test_pp2_tsv)
        elif 'TEST' in data_set:
            if n == 50 and ('train' in data_set or 'dev' in data_set):
                pd.read_pickle(top_50_sbert_sims_training_TEST_df).to_csv(top_50_sbert_sims_training_TEST_tsv)
                pd.read_pickle(top_50_infersent_sims_training_TEST_df).to_csv(top_50_infersent_sims_training_TEST_tsv)
                pd.read_pickle(top_50_universal_sims_training_TEST_df).to_csv(top_50_universal_sims_training_TEST_tsv)
                pd.read_pickle(top_50_sim_cse_sims_training_TEST_df).to_csv(top_50_sim_cse_sims_training_TEST_tsv)
            if n == 50 and 'test' in data_set:
                pd.read_pickle(top_50_sbert_sims_test_TEST_df).to_csv(top_50_sbert_sims_test_TEST_tsv)
                pd.read_pickle(top_50_infersent_sims_test_TEST_df).to_csv(top_50_infersent_sims_test_TEST_tsv)
                pd.read_pickle(top_50_universal_sims_test_TEST_df).to_csv(top_50_universal_sims_test_TEST_tsv)
                pd.read_pickle(top_50_sim_cse_sims_test_TEST_df).to_csv(top_50_sim_cse_sims_test_TEST_tsv)
        else:
            if n == 50 and ('train' in data_set or 'dev' in data_set):
                pd.read_pickle(top_50_sbert_sims_training_pp1_df).to_csv(top_50_sbert_sims_training_pp1_tsv)
                pd.read_pickle(top_50_infersent_sims_training_pp1_df).to_csv(top_50_infersent_sims_training_pp1_tsv)
                pd.read_pickle(top_50_universal_sims_training_pp1_df).to_csv(top_50_universal_sims_training_pp1_tsv)
                pd.read_pickle(top_50_sim_cse_sims_training_pp1_df).to_csv(top_50_sim_cse_sims_training_pp1_tsv)
            if n == 50 and 'test' in data_set:
                pd.read_pickle(top_50_sbert_sims_test_pp1_df).to_csv(top_50_sbert_sims_test_pp1_tsv)
                pd.read_pickle(top_50_infersent_sims_test_pp1_df).to_csv(top_50_infersent_sims_test_pp1_tsv)
                pd.read_pickle(top_50_universal_sims_test_pp1_df).to_csv(top_50_universal_sims_test_pp1_tsv)
                pd.read_pickle(top_50_sim_cse_sims_test_pp1_df).to_csv(top_50_sim_cse_sims_test_pp1_tsv)

    @staticmethod
    def create_pair_similarity_features(list_of_features, data_set):
        if Features.sbert.name in list_of_features:
            try:
                encodings_vclaims = sbert_encodings_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    encodings = sbert_encodings_training_pp1
                    filename = sbert_sims_training_pp1
                    filename_tsv = sbert_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = sbert_encodings_training_pp2
                        filename = sbert_sims_training_pp2
                        filename_tsv = sbert_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = sbert_encodings_training_TEST
                        filename = sbert_sims_training_TEST
                        filename_tsv = sbert_sims_training_TEST_tsv
                elif 'test' in data_set:
                    encodings = sbert_encodings_test_pp1
                    filename = sbert_sims_test_pp1
                    filename_tsv = sbert_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = sbert_encodings_test_pp2
                        filename = sbert_sims_test_pp2
                        filename_tsv = sbert_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = sbert_encodings_test_TEST
                        filename = sbert_sims_test_TEST
                        filename_tsv = sbert_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('cosine_sim')
                sim_score_data_generator.generate_all_sim_scores(encodings, encodings_vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing similarities for sbert.')
        if Features.infersent.name in list_of_features:
            try:
                encodings_vclaims = infersent_encodings_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    encodings = infersent_encodings_training_pp1
                    filename = infersent_sims_training_pp1
                    filename_tsv = infersent_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = infersent_encodings_training_pp2
                        filename = infersent_sims_training_pp2
                        filename_tsv = infersent_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = infersent_encodings_training_TEST
                        filename = infersent_sims_training_TEST
                        filename_tsv = infersent_sims_training_TEST_tsv
                if 'test' in data_set:
                    encodings = infersent_encodings_test_pp1
                    filename = infersent_sims_test_pp1
                    filename_tsv = infersent_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = infersent_encodings_test_pp2
                        filename = infersent_sims_test_pp2
                        filename_tsv = infersent_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = infersent_encodings_test_TEST
                        filename = infersent_sims_test_TEST
                        filename_tsv = infersent_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('cosine_sim')
                sim_score_data_generator.generate_all_sim_scores(encodings, encodings_vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing similarities for infersent.')
        if Features.universal.name in list_of_features:
            try:
                encodings_vclaims = universal_encodings_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    encodings = universal_encodings_training_pp1
                    filename = universal_sims_training_pp1
                    filename_tsv = universal_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = universal_encodings_training_pp2
                        filename = universal_sims_training_pp2
                        filename_tsv = universal_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = universal_encodings_training_TEST
                        filename = universal_sims_training_TEST
                        filename_tsv = universal_sims_training_TEST_tsv
                if 'test' in data_set:
                    encodings = universal_encodings_test_pp1
                    filename = universal_sims_test_pp1
                    filename_tsv = universal_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = universal_encodings_test_pp2
                        filename = universal_sims_test_pp2
                        filename_tsv = universal_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = universal_encodings_test_TEST
                        filename = universal_sims_test_TEST
                        filename_tsv = universal_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('cosine_sim')
                sim_score_data_generator.generate_all_sim_scores(encodings, encodings_vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing similarities for universal sentence encoder.')
        if Features.sim_cse.name in list_of_features:
            try:
                encodings_vclaims = sim_cse_encodings_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    encodings = sim_cse_encodings_training_pp1
                    filename = sim_cse_sims_training_pp1
                    filename_tsv = sim_cse_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = sim_cse_encodings_training_pp2
                        filename = sim_cse_sims_training_pp2
                        filename_tsv = sim_cse_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = sim_cse_encodings_training_TEST
                        filename = sim_cse_sims_training_TEST
                        filename_tsv = sim_cse_sims_training_TEST_tsv
                if 'test' in data_set:
                    encodings = sim_cse_encodings_test_pp1
                    filename = sim_cse_sims_test_pp1
                    filename_tsv = sim_cse_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        encodings = sim_cse_encodings_test_pp2
                        filename = sim_cse_sims_test_pp2
                        filename_tsv = sim_cse_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        encodings = sim_cse_encodings_test_TEST
                        filename = sim_cse_sims_test_TEST
                        filename_tsv = sim_cse_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('cosine_sim')
                sim_score_data_generator.generate_all_sim_scores(encodings, encodings_vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing similarities for sim cse encoder.')
        if Features.seq_match.name in list_of_features:
            try:
                vclaims = v_claims_df
                if 'train' in data_set or 'dev' in data_set:
                    filename = seq_match_training_pp1
                    filename_tsv = seq_match_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = seq_match_training_pp2
                        filename_tsv = seq_match_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = seq_match_training_TEST
                        filename_tsv = seq_match_training_TEST_tsv
                if 'test' in data_set:
                    filename = seq_match_test_pp1
                    filename_tsv = seq_match_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = seq_match_test_pp2
                        filename_tsv = seq_match_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = seq_match_test_TEST
                        filename_tsv = seq_match_test_TEST_tsv
                sim_score_generator = SimilarityScoreDataGenerator('sequence_matching_sim')
                sim_score_generator.generate_all_sim_scores(data_set, vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing sequence matching similarity.')
        if Features.levenshtein.name in list_of_features:
            try:
                vclaims = v_claims_df
                if 'train' in data_set or 'dev' in data_set:
                    filename = levenshtein_training_pp1
                    filename_tsv = levenshtein_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = levenshtein_training_pp2
                        filename_tsv = levenshtein_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = levenshtein_training_TEST
                        filename_tsv = levenshtein_training_TEST_tsv
                if 'test' in data_set:
                    filename = levenshtein_test_pp1
                    filename_tsv = levenshtein_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = levenshtein_test_pp2
                        filename_tsv = levenshtein_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = levenshtein_test_TEST
                        filename_tsv = levenshtein_test_TEST_tsv
                sim_score_generator = SimilarityScoreDataGenerator('levenshtein_dist')
                sim_score_generator.generate_all_sim_scores(data_set, vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing levenshtein distance.')
        if Features.jacc_chars.name in list_of_features:
            try:
                vclaims = v_claims_df
                if 'train' in data_set or 'dev' in data_set:
                    filename = jacc_chars_training_pp1
                    filename_tsv = jacc_chars_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = jacc_chars_training_pp2
                        filename_tsv = jacc_chars_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = jacc_chars_training_TEST
                        filename_tsv = jacc_chars_training_TEST_tsv
                if 'test' in data_set:
                    filename = jacc_chars_test_pp1
                    filename_tsv = jacc_chars_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = jacc_chars_test_pp2
                        filename_tsv = jacc_chars_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = jacc_chars_test_TEST
                        filename_tsv = jacc_chars_test_TEST_tsv
                sim_score_generator = SimilarityScoreDataGenerator('jacquard_dist')
                sim_score_generator.generate_all_sim_scores(data_set, vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing jaccard distance for characters.')
        if Features.jacc_tokens.name in list_of_features:
            try:
                vclaims = v_claims_df
                if 'train' in data_set or 'dev' in data_set:
                    filename = jacc_tokens_training_pp1
                    filename_tsv = jacc_tokens_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = jacc_tokens_training_pp2
                        filename_tsv = jacc_tokens_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = jacc_tokens_training_TEST
                        filename_tsv = jacc_tokens_training_TEST_tsv
                if 'test' in data_set:
                    filename = jacc_tokens_test_pp1
                    filename_tsv = jacc_tokens_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = jacc_tokens_test_pp2
                        filename_tsv = jacc_tokens_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = jacc_tokens_test_TEST
                        filename_tsv = jacc_tokens_test_TEST_tsv
                sim_score_generator = SimilarityScoreDataGenerator('jacquard_dist_token')
                sim_score_generator.generate_all_sim_scores(data_set, vclaims, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong computing jaccard distance for tokens.')
        if Features.ne.name in list_of_features:
            try:
                vclaims_ne = ne_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    nes = ne_training_pp1
                    filename = ne_sims_training_pp1
                    filename_tsv = ne_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        nes = ne_training_pp2
                        filename = ne_sims_training_pp2
                        filename_tsv = ne_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        nes = ne_training_TEST
                        filename = ne_sims_training_TEST
                        filename_tsv = ne_sims_training_TEST_tsv
                elif 'test' in data_set:
                    nes = ne_test_pp1
                    filename = ne_sims_test_pp1
                    filename_tsv = ne_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        nes = ne_test_pp2
                        filename = ne_sims_test_pp2
                        filename_tsv = ne_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        nes = ne_test_TEST
                        filename = ne_sims_test_TEST
                        filename_tsv = ne_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('ne_sim')
                sim_score_data_generator.generate_all_sim_scores(nes, vclaims_ne, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing named entity similarity.')
        if Features.main_syms.name in list_of_features:
            try:
                vclaims_main_syms = main_syms_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    main_syms = main_syms_training_pp1
                    filename = main_syms_sims_training_pp1
                    filename_tsv = main_syms_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        main_syms = main_syms_training_pp2
                        filename = main_syms_sims_training_pp2
                        filename_tsv = main_syms_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        main_syms = main_syms_training_TEST
                        filename = main_syms_sims_training_TEST
                        filename_tsv = main_syms_sims_training_TEST_tsv
                elif 'test' in data_set:
                    main_syms = main_syms_test_pp1
                    filename = main_syms_sims_test_pp1
                    filename_tsv = main_syms_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        main_syms = main_syms_test_pp2
                        filename = main_syms_sims_test_pp2
                        filename_tsv = main_syms_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        main_syms = main_syms_test_TEST
                        filename = main_syms_sims_test_TEST
                        filename_tsv = main_syms_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('syn_sim')
                sim_score_data_generator.generate_all_sim_scores(main_syms, vclaims_main_syms, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing main synonym similarity.')
        if Features.words.name in list_of_features:
            try:
                vclaims_words = words_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    words = words_training_pp1
                    filename = words_sims_training_pp1
                    filename_tsv = words_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        words = words_training_pp2
                        filename = words_sims_training_pp2
                        filename_tsv = words_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        words = words_training_TEST
                        filename = words_sims_training_TEST
                        filename_tsv = words_sims_training_TEST_tsv
                elif 'test' in data_set:
                    words = words_test_pp1
                    filename = words_sims_test_pp1
                    filename_tsv = words_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        words = words_test_pp2
                        filename = words_sims_test_pp2
                        filename_tsv = words_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        words = words_test_TEST
                        filename = words_sims_test_TEST
                        filename_tsv = words_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('syn_sim')
                sim_score_data_generator.generate_all_sim_scores(words, vclaims_words, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing word similarity.')
        if Features.subjects.name in list_of_features:
            try:
                vclaims_subjects = subjects_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    subjects = subjects_training_pp1
                    filename = subjects_sims_training_pp1
                    filename_tsv = subjects_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        subjects = subjects_training_pp2
                        filename = subjects_sims_training_pp2
                        filename_tsv = subjects_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        subjects = subjects_training_TEST
                        filename = subjects_sims_training_TEST
                        filename_tsv = subjects_sims_training_TEST_tsv
                elif 'test' in data_set:
                    subjects = subjects_test_pp1
                    filename = subjects_sims_test_pp1
                    filename_tsv = subjects_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        subjects = subjects_test_pp2
                        filename = subjects_sims_test_pp2
                        filename_tsv = subjects_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        subjects = subjects_test_TEST
                        filename = subjects_sims_test_TEST
                        filename_tsv = subjects_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('subject_sim')
                sim_score_data_generator.generate_all_sim_scores(subjects, vclaims_subjects, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing subject similarity.')
        if Features.token_number.name in list_of_features:
            try:
                vclaims_token_number = token_number_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    token_numbers = token_number_training_pp1
                    filename = token_number_sims_training_pp1
                    filename_tsv = token_number_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        token_numbers = token_number_training_pp2
                        filename = token_number_sims_training_pp2
                        filename_tsv = token_number_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        token_numbers = token_number_training_TEST
                        filename = token_number_sims_training_TEST
                        filename_tsv = token_number_sims_training_TEST_tsv
                elif 'test' in data_set:
                    token_numbers = token_number_test_pp1
                    filename = token_number_sims_test_pp1
                    filename_tsv = token_number_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        token_numbers = token_number_test_pp2
                        filename = token_number_sims_test_pp2
                        filename_tsv = token_number_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        token_numbers = token_number_test_TEST
                        filename = token_number_sims_test_TEST
                        filename_tsv = token_number_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('token_number_sim')
                sim_score_data_generator.generate_all_sim_scores(token_numbers, vclaims_token_number, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing Token Number Similarity.')
        if Features.ne_ne_ratio.name in list_of_features:
            try:
                vclaims_ne = ne_vclaims_pp1
                if 'train' in data_set or 'dev' in data_set:
                    nes = ne_training_pp1
                    filename = ne_ne_ratio_sims_training_pp1
                    filename_tsv = ne_ne_ratio_sims_training_pp1_tsv
                    if 'pp2' in data_set:
                        nes = ne_training_pp2
                        filename = ne_ne_ratio_sims_training_pp2
                        filename_tsv = ne_ne_ratio_sims_training_pp2_tsv
                    elif 'TEST' in data_set:
                        nes = ne_training_TEST
                        filename = ne_ne_ratio_sims_training_TEST
                        filename_tsv = ne_ne_ratio_sims_training_TEST_tsv
                elif 'test' in data_set:
                    nes = ne_test_pp1
                    filename = ne_ne_ratio_sims_test_pp1
                    filename_tsv = ne_ne_ratio_sims_test_pp1_tsv
                    if 'pp2' in data_set:
                        nes = ne_test_pp2
                        filename = ne_ne_ratio_sims_test_pp2
                        filename_tsv = ne_ne_ratio_sims_test_pp2_tsv
                    elif 'TEST' in data_set:
                        nes = ne_test_TEST
                        filename = ne_ne_ratio_sims_test_TEST
                        filename_tsv = ne_ne_ratio_sims_test_TEST_tsv
                sim_score_data_generator = SimilarityScoreDataGenerator('ne_ne_ratio_sim')
                sim_score_data_generator.generate_all_sim_scores(nes, vclaims_ne, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with computing NE-NE-Ratio.')
