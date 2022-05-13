import pandas as pd
from src.feature_generation import top_50_sbert_sims_training_pp1_df, sbert_sims_training_pp1, \
    top_50_infersent_sims_training_pp1_df, infersent_sims_training_pp1, train_sbert_infersent_disjunction, \
    sbert_sims_test_pp1, top_50_sbert_sims_test_pp1_df, top_50_infersent_sims_test_pp1_df, infersent_sims_test_pp1, \
    test_sbert_infersent_disjunction, top_50_universal_sims_training_pp1_df, universal_sims_training_pp1, \
    train_sbert_infersent_universal_disjunction, top_50_universal_sims_test_pp1_df, universal_sims_test_pp1, \
    test_sbert_infersent_universal_disjunction, top_50_sim_cse_sims_training_pp1_df, sim_cse_sims_training_pp1, \
    top_50_sim_cse_sims_test_pp1_df, sim_cse_sims_test_pp1, train_sbert_infersent_universal_sim_cse_disjunction, \
    test_sbert_infersent_universal_sim_cse_disjunction, train_sbert_infersent_universal_sim_cse_disjunction_tsv, \
    test_sbert_infersent_universal_sim_cse_disjunction_tsv, seq_match_training_pp1, levenshtein_training_pp1, \
    jacc_chars_training_pp1, jacc_tokens_training_pp1, ne_sims_training_pp1, main_syms_sims_training_pp1, \
    words_sims_training_pp1, subjects_sims_training_pp1, train_first_five_features, train_first_six_features, \
    train_first_seven_features, train_first_eight_features, train_first_nine_features, train_first_ten_features, \
    train_first_eleven_features, train_first_twelve_features, seq_match_test_pp1, levenshtein_test_pp1, \
    jacc_chars_test_pp1, jacc_tokens_test_pp1, ne_sims_test_pp1, main_syms_sims_test_pp1, words_sims_test_pp1, \
    subjects_sims_test_pp1, test_first_five_features, test_first_six_features, test_first_seven_features, \
    test_first_eight_features, test_first_nine_features, test_first_ten_features, test_first_eleven_features, \
    complete_feature_set_pairs_train, complete_feature_set_pairs_train_tsv, words_token_ratio_sims_training_pp1, \
    main_syms_token_ratio_sims_training_pp1, \
    words_ratio_sims_training_pp1, main_syms_ratio_sims_training_pp1, ne_token_ratio_sims_training_pp1, \
    ne_ne_ratio_sims_training_pp1, token_number_sims_training_pp1, train_first_thirteen_features, \
    train_first_fourteen_features, train_first_fifteen_features, train_first_sixteen_features, \
    train_first_seventeen_features, train_first_eighteen_features, train_first_nineteen_features, \
    token_number_sims_test_pp1, ne_ne_ratio_sims_test_pp1, ne_token_ratio_sims_test_pp1, main_syms_ratio_sims_test_pp1, \
    main_syms_token_ratio_sims_test_pp1, words_ratio_sims_test_pp1, words_token_ratio_sims_test_pp1, \
    test_first_twelve_features, test_first_thirteen_features, test_first_fourteen_features, test_first_fifteen_features, \
    test_first_sixteen_features, test_first_eighteen_features, test_first_seventeen_features, \
    test_first_nineteen_features, train_first_nineteen_features_tsv, test_first_nineteen_features_tsv, \
    complete_feature_set_pairs_test, complete_feature_set_pairs_test_tsv
from src.feature_generation.file_paths.TEST_file_names import top_50_sbert_sims_training_TEST_df, \
    sbert_sims_training_TEST, top_50_infersent_sims_training_TEST_df, infersent_sims_training_TEST, \
    top_50_universal_sims_training_TEST_df, universal_sims_training_TEST, top_50_sim_cse_sims_training_TEST_df, \
    train_sbert_infersent_disjunction_TEST, sim_cse_sims_training_TEST, \
    train_sbert_infersent_universal_disjunction_TEST, train_sbert_infersent_universal_sim_cse_disjunction_TEST, \
    train_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST, top_50_sbert_sims_test_TEST_df, sbert_sims_test_TEST, \
    top_50_infersent_sims_test_TEST_df, infersent_sims_test_TEST, top_50_universal_sims_test_TEST_df, \
    universal_sims_test_TEST, top_50_sim_cse_sims_test_TEST_df, sim_cse_sims_test_TEST, \
    test_sbert_infersent_disjunction_TEST, test_sbert_infersent_universal_disjunction_TEST, \
    test_sbert_infersent_universal_sim_cse_disjunction_TEST, \
    test_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST, seq_match_training_TEST, levenshtein_training_TEST, \
    jacc_chars_training_TEST, jacc_tokens_training_TEST, ne_sims_training_TEST, words_sims_training_TEST, \
    subjects_sims_training_TEST, main_syms_sims_training_TEST, train_first_five_features_TEST, \
    train_first_six_features_TEST, train_first_seven_features_TEST, train_first_eight_features_TEST, \
    train_first_nine_features_TEST, train_first_ten_features_TEST, train_first_eleven_features_TEST, \
    train_first_twelve_features_TEST, seq_match_test_TEST, levenshtein_test_TEST, \
    jacc_chars_test_TEST, jacc_tokens_test_TEST, ne_sims_test_TEST, main_syms_sims_test_TEST, words_sims_test_TEST, \
    subjects_sims_test_TEST, test_first_five_features_TEST, test_first_six_features_TEST, \
    test_first_seven_features_TEST, test_first_eight_features_TEST, test_first_nine_features_TEST, \
    test_first_ten_features_TEST, test_first_eleven_features_TEST, complete_feature_set_pairs_train_TEST, \
    complete_feature_set_pairs_train_tsv_TEST, token_number_sims_training_TEST, ne_ne_ratio_sims_training_TEST, \
    ne_token_ratio_sims_training_TEST, main_syms_ratio_sims_training_TEST, main_syms_token_ratio_sims_training_TEST, \
    words_ratio_sims_training_TEST, words_token_ratio_sims_training_TEST, train_first_thirteen_features_TEST, \
    train_first_fourteen_features_TEST, train_first_fifteen_features_TEST, train_first_sixteen_features_TEST, \
    train_first_seventeen_features_TEST, train_first_eighteen_features_TEST, train_first_nineteen_features_TEST, \
    test_first_nineteen_features_TEST, test_first_eighteen_features_TEST, token_number_sims_test_TEST, \
    ne_ne_ratio_sims_test_TEST, ne_token_ratio_sims_test_TEST, main_syms_token_ratio_sims_test_TEST, \
    main_syms_ratio_sims_test_TEST, words_ratio_sims_test_TEST, words_token_ratio_sims_test_TEST, \
    test_first_twelve_features_TEST, test_first_thirteen_features_TEST, test_first_fourteen_features_TEST, \
    test_first_fifteen_features_TEST, test_first_sixteen_features_TEST, test_first_seventeen_features_TEST, \
    train_first_nineteen_features_TEST_tsv, test_first_nineteen_features_TEST_tsv, complete_feature_set_pairs_test_TEST, \
    complete_feature_set_pairs_test_tsv_TEST
from src.feature_generation.file_paths.pp2_files import train_sbert_infersent_universal_sim_cse_disjunction_pp2, \
    train_sbert_infersent_disjunction_pp2, sim_cse_sims_training_pp2, top_50_sim_cse_sims_training_pp2_df, \
    universal_sims_training_pp2, top_50_universal_sims_training_pp2_df, infersent_sims_training_pp2, \
    top_50_infersent_sims_training_pp2_df, sbert_sims_training_pp2, top_50_sbert_sims_training_pp2_df, \
    train_sbert_infersent_universal_disjunction_pp2, train_sbert_infersent_universal_sim_cse_disjunction_tsv_pp2, \
    top_50_sbert_sims_test_pp2_df, sbert_sims_test_pp2, top_50_infersent_sims_test_pp2_df, infersent_sims_test_pp2, \
    top_50_universal_sims_test_pp2_df, universal_sims_test_pp2, top_50_sim_cse_sims_test_pp2_df, sim_cse_sims_test_pp2, \
    test_sbert_infersent_disjunction_pp2, test_sbert_infersent_universal_disjunction_pp2, \
    test_sbert_infersent_universal_sim_cse_disjunction_pp2, test_sbert_infersent_universal_sim_cse_disjunction_tsv_pp2, \
    seq_match_training_pp2, levenshtein_training_pp2, jacc_chars_training_pp2, jacc_tokens_training_pp2, \
    ne_sims_training_pp2, main_syms_sims_training_pp2, words_sims_training_pp2, subjects_sims_training_pp2, \
    train_first_five_features_pp2, train_first_six_features_pp2, train_first_seven_features_pp2, \
    train_first_eight_features_pp2, train_first_nine_features_pp2, train_first_ten_features_pp2, \
    train_first_eleven_features_pp2, train_first_twelve_features_pp2, \
    test_first_eleven_features_pp2, \
    test_first_ten_features_pp2, test_first_nine_features_pp2, test_first_eight_features_pp2, \
    test_first_seven_features_pp2, test_first_six_features_pp2, test_first_five_features_pp2, seq_match_test_pp2, \
    levenshtein_test_pp2, jacc_chars_test_pp2, jacc_tokens_test_pp2, ne_sims_test_pp2, main_syms_sims_test_pp2, \
    words_sims_test_pp2, subjects_sims_test_pp2, complete_feature_set_pairs_train_pp2, \
    complete_feature_set_pairs_train_tsv_pp2, token_number_sims_training_pp2, ne_token_ratio_sims_training_pp2, \
    ne_ne_ratio_sims_training_pp2, main_syms_token_ratio_sims_training_pp2, main_syms_ratio_sims_training_pp2, \
    words_ratio_sims_training_pp2, words_token_ratio_sims_training_pp2, train_first_thirteen_features_pp2, \
    train_first_fourteen_features_pp2, train_first_fifteen_features_pp2, train_first_sixteen_features_pp2, \
    train_first_seventeen_features_pp2, train_first_nineteen_features_pp2, train_first_eighteen_features_pp2, \
    token_number_sims_test_pp2, ne_ne_ratio_sims_test_pp2, ne_token_ratio_sims_test_pp2, main_syms_ratio_sims_test_pp2, \
    main_syms_token_ratio_sims_test_pp2, words_ratio_sims_test_pp2, words_token_ratio_sims_test_pp2, \
    test_first_twelve_features_pp2, test_first_thirteen_features_pp2, test_first_fourteen_features_pp2, \
    test_first_sixteen_features_pp2, test_first_fifteen_features_pp2, test_first_seventeen_features_pp2, \
    test_first_nineteen_features_pp2, test_first_eighteen_features_pp2, train_first_nineteen_features_pp2_tsv, \
    test_first_nineteen_features_pp2_tsv, complete_feature_set_pairs_test_pp2, complete_feature_set_pairs_test_tsv_pp2
from src.feature_generation.src.creating_datafiles.feature_set_maker import FeatureSetMaker


class FeatureSetCombiner:

    @staticmethod
    def combine_top_50_sentence_embeddings_features(data_set):
            if 'train' in data_set or 'dev' in data_set:
                file_1 = top_50_sbert_sims_training_pp1_df
                file_1_extended = sbert_sims_training_pp1
                file_2 = top_50_infersent_sims_training_pp1_df
                file_2_extended = infersent_sims_training_pp1
                file_3 = top_50_universal_sims_training_pp1_df
                file_3_extended = universal_sims_training_pp1
                file_4 = top_50_sim_cse_sims_training_pp1_df
                file_4_extended = sim_cse_sims_training_pp1
                two_output_df = train_sbert_infersent_disjunction
                three_output_df = train_sbert_infersent_universal_disjunction
                four_output_df = train_sbert_infersent_universal_sim_cse_disjunction
                four_output_tsv = train_sbert_infersent_universal_sim_cse_disjunction_tsv
                if 'pp2' in data_set:
                    file_1 = top_50_sbert_sims_training_pp2_df
                    file_1_extended = sbert_sims_training_pp2
                    file_2 = top_50_infersent_sims_training_pp2_df
                    file_2_extended = infersent_sims_training_pp2
                    file_3 = top_50_universal_sims_training_pp2_df
                    file_3_extended = universal_sims_training_pp2
                    file_4 = top_50_sim_cse_sims_training_pp2_df
                    file_4_extended = sim_cse_sims_training_pp2
                    two_output_df = train_sbert_infersent_disjunction_pp2
                    three_output_df = train_sbert_infersent_universal_disjunction_pp2
                    four_output_df = train_sbert_infersent_universal_sim_cse_disjunction_pp2
                    four_output_tsv = train_sbert_infersent_universal_sim_cse_disjunction_tsv_pp2
                elif 'TEST' in data_set:
                    file_1 = top_50_sbert_sims_training_TEST_df
                    file_1_extended = sbert_sims_training_TEST
                    file_2 = top_50_infersent_sims_training_TEST_df
                    file_2_extended = infersent_sims_training_TEST
                    file_3 = top_50_universal_sims_training_TEST_df
                    file_3_extended = universal_sims_training_TEST
                    file_4 = top_50_sim_cse_sims_training_TEST_df
                    file_4_extended = sim_cse_sims_training_TEST
                    two_output_df = train_sbert_infersent_disjunction_TEST
                    three_output_df = train_sbert_infersent_universal_disjunction_TEST
                    four_output_df = train_sbert_infersent_universal_sim_cse_disjunction_TEST
                    four_output_tsv = train_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST
            elif 'test' in data_set:
                file_1 = top_50_sbert_sims_test_pp1_df
                file_1_extended = sbert_sims_test_pp1
                file_2 = top_50_infersent_sims_test_pp1_df
                file_2_extended = infersent_sims_test_pp1
                file_3 = top_50_universal_sims_test_pp1_df
                file_3_extended = universal_sims_test_pp1
                file_4 = top_50_sim_cse_sims_test_pp1_df
                file_4_extended = sim_cse_sims_test_pp1
                two_output_df = test_sbert_infersent_disjunction
                three_output_df = test_sbert_infersent_universal_disjunction
                four_output_df = test_sbert_infersent_universal_sim_cse_disjunction
                four_output_tsv = test_sbert_infersent_universal_sim_cse_disjunction_tsv
                if 'pp2' in data_set:
                    file_1 = top_50_sbert_sims_test_pp2_df
                    file_1_extended = sbert_sims_test_pp2
                    file_2 = top_50_infersent_sims_test_pp2_df
                    file_2_extended = infersent_sims_test_pp2
                    file_3 = top_50_universal_sims_test_pp2_df
                    file_3_extended = universal_sims_test_pp2
                    file_4 = top_50_sim_cse_sims_test_pp2_df
                    file_4_extended = sim_cse_sims_test_pp2
                    two_output_df = test_sbert_infersent_disjunction_pp2
                    three_output_df = test_sbert_infersent_universal_disjunction_pp2
                    four_output_df = test_sbert_infersent_universal_sim_cse_disjunction_pp2
                    four_output_tsv = test_sbert_infersent_universal_sim_cse_disjunction_tsv_pp2
                elif 'TEST' in data_set:
                    file_1 = top_50_sbert_sims_test_TEST_df
                    file_1_extended = sbert_sims_test_TEST
                    file_2 = top_50_infersent_sims_test_TEST_df
                    file_2_extended = infersent_sims_test_TEST
                    file_3 = top_50_universal_sims_test_TEST_df
                    file_3_extended = universal_sims_test_TEST
                    file_4 = top_50_sim_cse_sims_test_TEST_df
                    file_4_extended = sim_cse_sims_test_TEST
                    two_output_df = test_sbert_infersent_disjunction_TEST
                    three_output_df = test_sbert_infersent_universal_disjunction_TEST
                    four_output_df = test_sbert_infersent_universal_sim_cse_disjunction_TEST
                    four_output_tsv = test_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST
            FeatureSetMaker.combine_feature_dataframes_disjunction(data_set, file_1, file_1_extended,
                                                                   file_2, file_2_extended, two_output_df)
            FeatureSetMaker.combine_three_feature_dataframes_disjunction(data_set, two_output_df, file_1_extended,
                                                         file_2_extended, file_3, file_3_extended, three_output_df)
            FeatureSetMaker.combine_four_feature_dataframes_disjunction(data_set, three_output_df, file_1_extended,
                                                        file_2_extended, file_3_extended, file_4, file_4_extended,
                                                        four_output_df)
            pd.read_pickle(four_output_df).to_csv(four_output_tsv)

    @staticmethod
    def add_other_features_to_embedding_features(data_set, list_of_features):
        if list_of_features == ['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
                               'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
                               'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
                               'words_token_ratio']:
            if 'train' in data_set or 'dev' in data_set:
                feature_set_base = train_sbert_infersent_universal_sim_cse_disjunction
                features_five = seq_match_training_pp1
                features_six = levenshtein_training_pp1
                features_seven = jacc_chars_training_pp1
                features_eight = jacc_tokens_training_pp1
                features_nine = ne_sims_training_pp1
                features_ten = main_syms_sims_training_pp1
                features_eleven = words_sims_training_pp1
                features_twelve = subjects_sims_training_pp1
                features_thirteen = token_number_sims_training_pp1
                features_fourteen = ne_ne_ratio_sims_training_pp1
                features_fifteen = ne_token_ratio_sims_training_pp1
                features_sixteen = main_syms_ratio_sims_training_pp1
                features_seventeen = main_syms_token_ratio_sims_training_pp1
                features_eighteen = words_ratio_sims_training_pp1
                features_nineteen = words_token_ratio_sims_training_pp1
                feature_set_five = train_first_five_features
                feature_set_six = train_first_six_features
                feature_set_seven = train_first_seven_features
                feature_set_eight = train_first_eight_features
                feature_set_nine = train_first_nine_features
                feature_set_ten = train_first_ten_features
                feature_set_eleven = train_first_eleven_features
                feature_set_twelve = train_first_twelve_features
                feature_set_thirteen = train_first_thirteen_features
                feature_set_fourteen = train_first_fourteen_features
                feature_set_fifteen = train_first_fifteen_features
                feature_set_sixteen = train_first_sixteen_features
                feature_set_seventeen = train_first_seventeen_features
                feature_set_eighteen = train_first_eighteen_features
                feature_set_nineteen = train_first_nineteen_features
                feature_set_nineteen_tsv = train_first_nineteen_features_tsv
                if 'pp2' in data_set:
                    feature_set_base = train_sbert_infersent_universal_sim_cse_disjunction_pp2
                    features_five = seq_match_training_pp2
                    features_six = levenshtein_training_pp2
                    features_seven = jacc_chars_training_pp2
                    features_eight = jacc_tokens_training_pp2
                    features_nine = ne_sims_training_pp2
                    features_ten = main_syms_sims_training_pp2
                    features_eleven = words_sims_training_pp2
                    features_twelve = subjects_sims_training_pp2
                    features_thirteen = token_number_sims_training_pp2
                    features_fourteen = ne_ne_ratio_sims_training_pp2
                    features_fifteen = ne_token_ratio_sims_training_pp2
                    features_sixteen = main_syms_ratio_sims_training_pp2
                    features_seventeen = main_syms_token_ratio_sims_training_pp2
                    features_eighteen = words_ratio_sims_training_pp2
                    features_nineteen = words_token_ratio_sims_training_pp2
                    feature_set_five = train_first_five_features_pp2
                    feature_set_six = train_first_six_features_pp2
                    feature_set_seven = train_first_seven_features_pp2
                    feature_set_eight = train_first_eight_features_pp2
                    feature_set_nine = train_first_nine_features_pp2
                    feature_set_ten = train_first_ten_features_pp2
                    feature_set_eleven = train_first_eleven_features_pp2
                    feature_set_twelve = train_first_twelve_features_pp2
                    feature_set_thirteen = train_first_thirteen_features_pp2
                    feature_set_fourteen = train_first_fourteen_features_pp2
                    feature_set_fifteen = train_first_fifteen_features_pp2
                    feature_set_sixteen = train_first_sixteen_features_pp2
                    feature_set_seventeen = train_first_seventeen_features_pp2
                    feature_set_eighteen = train_first_eighteen_features_pp2
                    feature_set_nineteen = train_first_nineteen_features_pp2
                    feature_set_nineteen_tsv = train_first_nineteen_features_pp2_tsv
                elif 'TEST' in data_set:
                    feature_set_base = train_sbert_infersent_universal_sim_cse_disjunction_TEST
                    features_five = seq_match_training_TEST
                    features_six = levenshtein_training_TEST
                    features_seven = jacc_chars_training_TEST
                    features_eight = jacc_tokens_training_TEST
                    features_nine = ne_sims_training_TEST
                    features_ten = main_syms_sims_training_TEST
                    features_eleven = words_sims_training_TEST
                    features_twelve = subjects_sims_training_TEST
                    features_thirteen = token_number_sims_training_TEST
                    features_fourteen = ne_ne_ratio_sims_training_TEST
                    features_fifteen = ne_token_ratio_sims_training_TEST
                    features_sixteen = main_syms_ratio_sims_training_TEST
                    features_seventeen = main_syms_token_ratio_sims_training_TEST
                    features_eighteen = words_ratio_sims_training_TEST
                    features_nineteen = words_token_ratio_sims_training_TEST
                    feature_set_five = train_first_five_features_TEST
                    feature_set_six = train_first_six_features_TEST
                    feature_set_seven = train_first_seven_features_TEST
                    feature_set_eight = train_first_eight_features_TEST
                    feature_set_nine = train_first_nine_features_TEST
                    feature_set_ten = train_first_ten_features_TEST
                    feature_set_eleven = train_first_eleven_features_TEST
                    feature_set_twelve = train_first_twelve_features_TEST
                    feature_set_thirteen = train_first_thirteen_features_TEST
                    feature_set_fourteen = train_first_fourteen_features_TEST
                    feature_set_fifteen = train_first_fifteen_features_TEST
                    feature_set_sixteen = train_first_sixteen_features_TEST
                    feature_set_seventeen = train_first_seventeen_features_TEST
                    feature_set_eighteen = train_first_eighteen_features_TEST
                    feature_set_nineteen = train_first_nineteen_features_TEST
                    feature_set_nineteen_tsv = train_first_nineteen_features_TEST_tsv
            elif 'test' in data_set:
                feature_set_base = test_sbert_infersent_universal_sim_cse_disjunction
                features_five = seq_match_test_pp1
                features_six = levenshtein_test_pp1
                features_seven = jacc_chars_test_pp1
                features_eight = jacc_tokens_test_pp1
                features_nine = ne_sims_test_pp1
                features_ten = main_syms_sims_test_pp1
                features_eleven = words_sims_test_pp1
                features_twelve = subjects_sims_test_pp1
                features_thirteen = token_number_sims_test_pp1
                features_fourteen = ne_ne_ratio_sims_test_pp1
                features_fifteen = ne_token_ratio_sims_test_pp1
                features_sixteen = main_syms_ratio_sims_test_pp1
                features_seventeen = main_syms_token_ratio_sims_test_pp1
                features_eighteen = words_ratio_sims_test_pp1
                features_nineteen = words_token_ratio_sims_test_pp1
                feature_set_five = test_first_five_features
                feature_set_six = test_first_six_features
                feature_set_seven = test_first_seven_features
                feature_set_eight = test_first_eight_features
                feature_set_nine = test_first_nine_features
                feature_set_ten = test_first_ten_features
                feature_set_eleven = test_first_eleven_features
                feature_set_twelve = test_first_twelve_features
                feature_set_thirteen = test_first_thirteen_features
                feature_set_fourteen = test_first_fourteen_features
                feature_set_fifteen = test_first_fifteen_features
                feature_set_sixteen = test_first_sixteen_features
                feature_set_seventeen = test_first_seventeen_features
                feature_set_eighteen = test_first_eighteen_features
                feature_set_nineteen = test_first_nineteen_features
                feature_set_nineteen_tsv = test_first_nineteen_features_tsv
                if 'pp2' in data_set:
                    feature_set_base = test_sbert_infersent_universal_sim_cse_disjunction_pp2
                    features_five = seq_match_test_pp2
                    features_six = levenshtein_test_pp2
                    features_seven = jacc_chars_test_pp2
                    features_eight = jacc_tokens_test_pp2
                    features_nine = ne_sims_test_pp2
                    features_ten = main_syms_sims_test_pp2
                    features_eleven = words_sims_test_pp2
                    features_twelve = subjects_sims_test_pp2
                    features_thirteen = token_number_sims_test_pp2
                    features_fourteen = ne_ne_ratio_sims_test_pp2
                    features_fifteen = ne_token_ratio_sims_test_pp2
                    features_sixteen = main_syms_ratio_sims_test_pp2
                    features_seventeen = main_syms_token_ratio_sims_test_pp2
                    features_eighteen = words_ratio_sims_test_pp2
                    features_nineteen = words_token_ratio_sims_test_pp2
                    feature_set_five = test_first_five_features_pp2
                    feature_set_six = test_first_six_features_pp2
                    feature_set_seven = test_first_seven_features_pp2
                    feature_set_eight = test_first_eight_features_pp2
                    feature_set_nine = test_first_nine_features_pp2
                    feature_set_ten = test_first_ten_features_pp2
                    feature_set_eleven = test_first_eleven_features_pp2
                    feature_set_twelve = test_first_twelve_features_pp2
                    feature_set_thirteen = test_first_thirteen_features_pp2
                    feature_set_fourteen = test_first_fourteen_features_pp2
                    feature_set_fifteen = test_first_fifteen_features_pp2
                    feature_set_sixteen = test_first_sixteen_features_pp2
                    feature_set_seventeen = test_first_seventeen_features_pp2
                    feature_set_eighteen = test_first_eighteen_features_pp2
                    feature_set_nineteen = test_first_nineteen_features_pp2
                    feature_set_nineteen_tsv = test_first_nineteen_features_pp2_tsv
                elif 'TEST' in data_set:
                    feature_set_base = test_sbert_infersent_universal_sim_cse_disjunction_TEST
                    features_five = seq_match_test_TEST
                    features_six = levenshtein_test_TEST
                    features_seven = jacc_chars_test_TEST
                    features_eight = jacc_tokens_test_TEST
                    features_nine = ne_sims_test_TEST
                    features_ten = main_syms_sims_test_TEST
                    features_eleven = words_sims_test_TEST
                    features_twelve = subjects_sims_test_TEST
                    features_thirteen = token_number_sims_test_TEST
                    features_fourteen = ne_ne_ratio_sims_test_TEST
                    features_fifteen = ne_token_ratio_sims_test_TEST
                    features_sixteen = main_syms_ratio_sims_test_TEST
                    features_seventeen = main_syms_token_ratio_sims_test_TEST
                    features_eighteen = words_ratio_sims_test_TEST
                    features_nineteen = words_token_ratio_sims_test_TEST
                    feature_set_five = test_first_five_features_TEST
                    feature_set_six = test_first_six_features_TEST
                    feature_set_seven = test_first_seven_features_TEST
                    feature_set_eight = test_first_eight_features_TEST
                    feature_set_nine = test_first_nine_features_TEST
                    feature_set_ten = test_first_ten_features_TEST
                    feature_set_eleven = test_first_eleven_features_TEST
                    feature_set_twelve = test_first_twelve_features_TEST
                    feature_set_thirteen = test_first_thirteen_features_TEST
                    feature_set_fourteen = test_first_fourteen_features_TEST
                    feature_set_fifteen = test_first_fifteen_features_TEST
                    feature_set_sixteen = test_first_sixteen_features_TEST
                    feature_set_seventeen = test_first_seventeen_features_TEST
                    feature_set_eighteen = test_first_eighteen_features_TEST
                    feature_set_nineteen = test_first_nineteen_features_TEST
                    feature_set_nineteen_tsv = test_first_nineteen_features_TEST_tsv
            FeatureSetMaker.add_features_to_dataset(feature_set_base, features_five, feature_set_five)
            FeatureSetMaker.add_features_to_dataset(feature_set_five, features_six, feature_set_six)
            FeatureSetMaker.add_features_to_dataset(feature_set_six, features_seven, feature_set_seven)
            FeatureSetMaker.add_features_to_dataset(feature_set_seven, features_eight, feature_set_eight)
            FeatureSetMaker.add_features_to_dataset(feature_set_eight, features_nine, feature_set_nine)
            FeatureSetMaker.add_features_to_dataset(feature_set_nine, features_ten, feature_set_ten)
            FeatureSetMaker.add_features_to_dataset(feature_set_ten, features_eleven, feature_set_eleven)
            FeatureSetMaker.add_features_to_dataset(feature_set_eleven, features_twelve, feature_set_twelve)
            FeatureSetMaker.add_features_to_dataset(feature_set_twelve, features_thirteen, feature_set_thirteen)
            FeatureSetMaker.add_features_to_dataset(feature_set_thirteen, features_fourteen, feature_set_fourteen)
            FeatureSetMaker.add_features_to_dataset(feature_set_fourteen, features_fifteen, feature_set_fifteen)
            FeatureSetMaker.add_features_to_dataset(feature_set_fifteen, features_sixteen, feature_set_sixteen)
            FeatureSetMaker.add_features_to_dataset(feature_set_sixteen, features_seventeen, feature_set_seventeen)
            FeatureSetMaker.add_features_to_dataset(feature_set_seventeen, features_eighteen, feature_set_eighteen)
            FeatureSetMaker.add_features_to_dataset(feature_set_eighteen, features_nineteen, feature_set_nineteen)
            pd.read_pickle(feature_set_nineteen).to_csv(feature_set_nineteen_tsv)
            df = pd.read_pickle(feature_set_nineteen)
            df.columns = ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
                               'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
                               'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
                               'words_token_ratio']
            if 'test' in data_set:
                complete_feature_set = complete_feature_set_pairs_test
                complete_feature_set_tsv = complete_feature_set_pairs_test_tsv
                if 'pp2' in data_set:
                    complete_feature_set = complete_feature_set_pairs_test_pp2
                    complete_feature_set_tsv = complete_feature_set_pairs_test_tsv_pp2
                elif 'TEST' in data_set:
                    complete_feature_set = complete_feature_set_pairs_test_TEST
                    complete_feature_set_tsv = complete_feature_set_pairs_test_tsv_TEST
            elif 'train' or 'dev' in data_set:
                complete_feature_set = complete_feature_set_pairs_train
                complete_feature_set_tsv = complete_feature_set_pairs_train_tsv
                if 'pp2' in data_set:
                    complete_feature_set = complete_feature_set_pairs_train_pp2
                    complete_feature_set_tsv = complete_feature_set_pairs_train_tsv_pp2
                elif 'TEST' in data_set:
                    complete_feature_set = complete_feature_set_pairs_train_TEST
                    complete_feature_set_tsv = complete_feature_set_pairs_train_tsv_TEST
            df.to_pickle(complete_feature_set)
            df.to_csv(complete_feature_set_tsv)

    @staticmethod
    def add_scores_to_feature_set(dataset, labels):
        if 'test' in dataset:
            pass
        elif 'train' or 'dev' in dataset:
            complete_feature_set = complete_feature_set_pairs_train
            complete_feature_set_tsv = complete_feature_set_pairs_train_tsv
            if 'pp2' in dataset:
                complete_feature_set = complete_feature_set_pairs_train_pp2
                complete_feature_set_tsv = complete_feature_set_pairs_train_tsv_pp2
            elif 'TEST' in dataset:
                complete_feature_set = complete_feature_set_pairs_train_TEST
                complete_feature_set_tsv = complete_feature_set_pairs_train_tsv_TEST
            complete_feature_set = FeatureSetMaker.add_correct_score_to_dataframe(complete_feature_set, labels,
                                                                                  complete_feature_set)
            complete_feature_set.to_csv(complete_feature_set_tsv)




