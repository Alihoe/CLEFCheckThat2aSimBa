import argparse

import pandas as pd

from src.evaluation.scorer.main import evaluate_CLEF
from src.feature_generation import pp_old_test_data, old_test_data, complete_feature_set_pairs_test, \
    complete_feature_set_pairs_train, old_predictions_binary, old_test_data_labels, pp_training_data, \
    all_training_data_labels
from src.feature_generation.feature_set_generator import FeatureSetGenerator
from src.feature_generation.file_paths.TEST_file_names import complete_feature_set_pairs_test_TEST
from src.feature_generation.unsupervised_feature_set_generator import UnsupervisedFeatureSetGenerator
from src.pre_processing import pre_processor
from src.prediction.feature_selector import FeatureSelector
from src.prediction.predictor import Predictor

top_5_sbert = 'data/unsupervised_ranking/pp1/top_5_sbert.tsv'
top_5_universal = 'data/unsupervised_ranking/pp1/top_5_universal.tsv'
top_5_infersent = 'data/unsupervised_ranking/pp1/top_5_infersent.tsv'
top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sim_cse.tsv'
top_5_seq_match = 'data/unsupervised_ranking/pp1/top_5_seq_match.tsv'
top_5_levenshtein = 'data/unsupervised_ranking/pp1/top_5_levenshtein.tsv'
top_5_jacc_chars = 'data/unsupervised_ranking/pp1/top_5_jacc_chars.tsv'
top_5_jacc_tokens = 'data/unsupervised_ranking/pp1/top_5_jacc_tokens.tsv'
top_5_ne = 'data/unsupervised_ranking/pp1/top_5_ne.tsv'
top_5_main_syms = 'data/unsupervised_ranking/pp1/top_5_main_syms.tsv'
top_5_words = 'data/unsupervised_ranking/pp1/top_5_words.tsv'
top_5_subjects = 'data/unsupervised_ranking/pp1/top_5_subjects.tsv'
top_5_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_ne_ne_ratio.tsv'
top_5_ne_token_ratio = 'data/unsupervised_ranking/pp1/top_5_ne_token_ratio.tsv'
top_5_main_syms_ratio = 'data/unsupervised_ranking/pp1/top_5_main_syms__ratio.tsv'
top_5_main_syms_token_ratio = 'data/unsupervised_ranking/pp1/top_5_main_syms_token_ratio.tsv'
top_5_words_ratio = 'data/unsupervised_ranking/pp1/top_5_words_ratio.tsv'
top_5_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_words_token_ratio.tsv'
top_5_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok.tsv'
top_5_sim_cse_jacc_tok_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_words.tsv'
top_5_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words.tsv'
top_5_sim_cse_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_ne.tsv'
top_5_sim_cse_jacc_tok_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_ne.tsv'
top_5_all_features = 'data/unsupervised_ranking/pp1/top_5_all_features.tsv'
top_5_all_features_without_infersent = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
top_5_no_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_no_sentence_embeddings.tsv'
top_5_sbert_universal_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse.tsv'
top_5_sbert_universal_sim_cse_ne_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_features.tsv'
top_5_sbert_universal_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_jacc_tok.tsv'
top_5_all_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_all_sentence_embeddings.tsv'
top_5_sbert_universal_sim_cse_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_ne_ratio_features.tsv'
top_5_sbert_universal_sim_cse_ne_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_main_syms_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_main_syms_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words.tsv'
top_5_sbert_infersent_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words.tsv'
top_5_sbert_infersent_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse.tsv'
top_5_sbert_infersent_sim_cse_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words_token_ratio.tsv'
top_5_sbert_sim_cse_features = 'data/unsupervised_ranking/pp1/top_5_sbert_sim_cse.tsv'
top_5_sbert_infersent_sim_cse_words_token_ratio_words = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent_sim_cse_words_token_ratio_words.tsv'
top_5_sbert_infersent = 'data/unsupervised_ranking/pp1/top_5_sbert_infersent.tsv'
top_5_sbert_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_sim_cse_words.tsv'

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Rank and score input claims.')
    # parser.add_argument('test_data', type=str, help='Pass test data that should be ranked.')
    # parser.add_argument('--preprocess', help='Preprocess the input tweets.')
    # parser.add_argument('-features', type=str, nargs='+',
    #                     default=['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein',
    #                              'jacc_chars', 'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number',
    #                              'ne_ne_ratio', 'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio',
    #                              'words_ratio', 'words_token_ratio'],
    #                     help='Pass a list of features that should be considered in the ranking.')
    # parser.add_argument('--unsupervised', help='Output an unsupervised ranking based on the selected features.')
    # parser.add_argument('--log_reg', help='Output a ranking based on a logistic regression classifier.')
    # parser.add_argument('--svm', help='Output a ranking based on a SVM classifier.')
    # parser.add_argument('--knn', help='Output a ranking based on a KNN classifier.')
    # parser.add_argument('--mn_naive_bayes', help='Output a ranking based on a naive bayes regression classifier.')
    # parser.add_argument('--dec_tree', help='Output a ranking based on a decision tree classifier.')
    # args = parser.parse_args()
    #
    # if args.preprocess:
    #     pre_processor.pre_process(args.test_data, pp_training_data)


    # FeatureSelector.feature_correlation(complete_feature_set_pairs_train, feature_correlation_training_data_spearman)
    # FeatureSelector.mutual_information_feature_selection(complete_feature_set_pairs_train)

    # pre_processor = PreProcessor('cleaning_tweets')
    # pp_training_data = pre_processor.pre_process(training_data, pp_training_data)
    # pp_old_test_data = pre_processor.pre_process(old_test_data, pp_old_test_data)

    # fsg = FeatureSetGenerator(['infersent'])
    # fsg.generate_feature_set(pp_old_test_data)

    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number'])
    # fsg.prepare_vclaims(v_claims)
    #
    # labels = fsg.combine_labels(training_data_labels_train, training_data_labels_dev, all_training_data_labels)
    # labels = all_training_data_labels
    # featureset_train = fsg.generate_feature_set(pp_training_data, labels)
    #
    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio'])
    #
    #
    # fsg.generate_feature_set(pp_training_data, all_training_data_labels)

    fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
                               'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
                               'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
                               'words_token_ratio'])

    pp2_test_data = 'data/TEST/pp2.tsv'

    fsg.generate_feature_set(pp2_test_data)


    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(complete_feature_set_pairs_train, complete_feature_set_pairs_test, old_test_data, old_predictions_binary)
    # evaluate_CLEF(old_test_data_labels, old_predictions_binary) #0.9163 without balancing
    #
    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'universal', 'sim_cse', 'score']]
    # test_df = pd.read_pickle(complete_feature_set_pairs_test)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'universal', 'sim_cse']]

    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(training_df, test_df, old_test_data, old_predictions_binary)
    # evaluate_CLEF(old_test_data_labels, old_predictions_binary) #0.9129 without balancing

    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio'])

    output = 'test.tsv'

    correlation = 'corr.tsv'
    new_test_data = 'data/pp_twitter_data/TEST/pp_test_TEST.tsv'

    test_data_labels = 'data/TEST/CT2022-Task2A-EN-Test_Qrels_gold.tsv'

    test_feature_set = 'data/feature_sets/test/TEST/complete_feature_set_pairs_test_TEST.pkl'

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # test_df = pd.read_pickle(test_feature_set)
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output) # log reg: 0.8832,
    # knn: 0.8865 ['sbert' 'infersent' 'sim_cse' 'words' 'words_token_ratio'],
    # mn naive bayes: 0.8793 ['sbert' 'infersent' 'sim_cse' 'words'],
    # dec_tree: 0.8502 ['sbert' 'infersent' 'sim_cse' 'words' 'words_token_ratio']
    # linear svc 0.8792  ['sbert' 'infersent' 'sim_cse' 'words' 'words_token_ratio']

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output) #0.8832

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output) #0.8760

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output) #0.8784

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'universal', 'sim_cse', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'universal', 'sim_cse']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8784

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8760

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8832
    #
    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8832


    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words', 'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words', 'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8832

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8832

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'ne_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'ne_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8820

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'ne_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'sim_cse', 'words_token_ratio', 'ne_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8820

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse', 'words', 'words_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse', 'words', 'words_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8820

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse', 'words', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'sim_cse', 'words']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8820

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words', 'jacc_tokens', 'ne_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words', 'jacc_tokens', 'ne_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8816

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words_token_ratio', 'jacc_tokens', 'ne_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words_token_ratio', 'jacc_tokens', 'ne_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8840

    # training_df = pd.read_pickle(complete_feature_set_pairs_train)
    # #training_df = training_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words_token_ratio', 'jacc_tokens', 'ne_token_ratio', 'score']]
    # test_df = pd.read_pickle(test_feature_set)
    # #test_df = test_df.loc[:, ['i_claim_id', 'ver_claim_id', 'sbert', 'words_token_ratio', 'jacc_tokens', 'ne_token_ratio']]
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(test_feature_set, training_df, test_df, new_test_data, output)
    # evaluate_CLEF(test_data_labels, output)  #0.8820



    # FeatureSelector.feature_correlation(test_feature_set, correlation)

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output)  #0.8632

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words', 'words_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output)  #0.9038

    ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words'], 'TEST')
    ufsg.create_top_n_output_file(new_test_data, output, n=5)
    evaluate_CLEF(test_data_labels, output)  #0.8884

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8801

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8644

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8896

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.9075

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.9143

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words', 'words_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.9127

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'words', 'jacc_tokens', 'ne_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8694

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'words_token_ratio', 'jacc_tokens', 'ne_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8679

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words_token_ratio', 'ne_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.9079

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein',
    #                                         'jacc_chars', 'jacc_tokens', 'ne', 'main_syms', 'words', 'ne_ne_ratio',
    #                                         'ne_token_ratio', 'main_syms_ratio',
    #                                         'main_syms_token_ratio', 'words_ratio', 'words_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4852

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.8711
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['infersent'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4208
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['universal'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.7153
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.7973
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['seq_match'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.2698
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['jacc_chars'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.0522
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['levenshtein'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.1271
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['jacc_tokens'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4014
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['ne'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4549
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.3228
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['words'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.5667
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['subjects'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.0608
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['ne_ne_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4357
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['ne_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.4620
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.3196
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.3071
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['words_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.6454
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['words_token_ratio'], 'TEST')
    # ufsg.create_top_n_output_file(new_test_data, output, n=5)
    # evaluate_CLEF(test_data_labels, output) #0.6630

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_infersent_sim_cse_words, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_infersent_sim_cse_words) # 0.9171

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_infersent_sim_cse, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_infersent_sim_cse) #0.9082

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_infersent_sim_cse_words_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_infersent_sim_cse_words_token_ratio) #0.9105

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words', 'words_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_infersent_sim_cse_words_token_ratio_words, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_infersent_sim_cse_words_token_ratio_words) #0.8804

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_infersent, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_infersent) #0.8973

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse) #0.8984????

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_sim_cse_features, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_sim_cse_features) #0.9162


    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_sim_cse_words, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_sim_cse_words) #0.9233

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'words_token'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_words_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_words_token_ratio) #0.8984

    # ufsg = UnsupervisedFeatureSetGenerator(['words_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_words_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_words_token_ratio) # 0.5488
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['words_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_words_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_words_ratio) # 0.5733

    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_main_syms_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_main_syms_token_ratio) #0.3192
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_main_syms_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_main_syms_ratio) #0.3780

    # ufsg = UnsupervisedFeatureSetGenerator(['ne_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_ne_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_ne_token_ratio) #0.2870

    # ufsg = UnsupervisedFeatureSetGenerator(['ne_ne_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_ne_ne_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_ne_ne_ratio) #0.3035

    # ufsg = UnsupervisedFeatureSetGenerator(['subjects'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_subjects, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_subjects) #0.0558

    # ufsg = UnsupervisedFeatureSetGenerator(['words'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_words, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_words) #0.5520

    # ufsg = UnsupervisedFeatureSetGenerator(['main_syms'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_main_syms, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_main_syms) #0.3111

    # ufsg = UnsupervisedFeatureSetGenerator(['ne'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_ne, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_ne) #0.3150

    # ufsg = UnsupervisedFeatureSetGenerator(['jacc_tokens'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_jacc_tokens, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_jacc_tokens) #0.4009
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['jacc_chars'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_jacc_chars, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_jacc_chars) #0.0502

    # ufsg = UnsupervisedFeatureSetGenerator(['levenshtein'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_levenshtein, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_levenshtein) #0.1696

    # ufsg = UnsupervisedFeatureSetGenerator(['seq_match'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_seq_match, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_seq_match) #0.2804

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_features, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_features) #0.8984????

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_all_sentence_embeddings, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_all_sentence_embeddings) #0.9064 infersent fast text

    # ufsg = UnsupervisedFeatureSetGenerator(['infersent'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_infersent, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_infersent) # Fast Text: 0.4644, Glove: 0.3468
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_features, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_features) # 0.9217

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'ne_ne_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_ne_ne_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_ne_ne_ratio) # 0.6899
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'ne_token_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_ne_token_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_ne_token_ratio) #  0.8915

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'main_syms_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_main_syms_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_main_syms_ratio) # 0.7368
    #
    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'words_ratio'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_words_ratio, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_words_ratio) # 0.8962

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert) # 0.8860 jetzt 0.8813?

    # ufsg = UnsupervisedFeatureSetGenerator(['sim_cse'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sim_cse, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sim_cse) # 0.8015

    # ufsg = UnsupervisedFeatureSetGenerator(['universal'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_universal, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_universal) # 0.7222

    #fsg.prepare_vclaims(v_claims)
    #
    # featureset_train = complete_feature_set_triples_train + '.pkl'
    # featureset_test = complete_feature_set_triples_test+'.pkl'
    #
    # predictor = Predictor('triple_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_triple)
    #
    # evaluate_CLEF(old_test_data_labels, old_predictions_triple) # MAP@5 = PRECISION@1 = MRR 0.8960
    #
    # featureset_train = complete_feature_set_pairs_train
    # featureset_test = complete_feature_set_pairs_test
    #
    # predictor = Predictor('highest_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_pedictions_highest_se_sims)
    # evaluate_CLEF(old_test_data_labels, old_pedictions_highest_se_sims) # MAP@5 = PRECISION@1 = MRR 0.8663
    #
    # predictor = Predictor('binary_proba')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_binary_proba)
    # evaluate_CLEF(old_test_data_labels, old_predictions_binary_proba) # MAP@5 = PRECISION@1 = MRR 0.8713
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_binary)
    # evaluate_CLEF(old_test_data_labels, old_predictions_binary) # MAP@5 = MRR 0.9053 PRECISION@1 = 0.8762
    #
    # OutputFormatter.drop_all_but_top_ver_claims(old_predictions_binary, old_predictions_binary_top_scores)
    # evaluate_CLEF(old_test_data_labels, old_predictions_binary_top_scores) # # MAP@5 = PRECISION@1 = MRR 0.8762
    #
    # featureset_train = complete_feature_set_triples_train+'.pkl'
    # featureset_test = complete_feature_set_triples_test+'.pkl'
    #
    # predictor = Predictor('triple_classification_with_rank_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_triple_double_classification)
    # evaluate_CLEF(old_test_data_labels, old_predictions_triple_double_classification)
    #
    #
    # featureset_train = complete_feature_set_pairs_train
    # featureset_test = complete_feature_set_pairs_test
    #
    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_highest_50_se_sims, n=50)
    # evaluate_CLEF(old_test_data_labels, old_predictions_highest_50_se_sims) #MAP@5 0.9081, PRECISION@1 0.8663 , MRR 0.9111
    #
    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_highest_10_se_sims, n=10)
    # evaluate_CLEF(old_test_data_labels, old_predictions_highest_10_se_sims) #MAP@5 0.9081, PRECISION@1 0.8663 , MRR  0.9107
    #
    # predictor = Predictor('highest_n_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, old_test_data, old_predictions_highest_5_se_sims)
    # evaluate_CLEF(old_test_data_labels, old_predictions_highest_5_se_sims) #MAP@5 0.9081, PRECISION@1 0.8663 , MRR 0.9081
    #
    # pd.read_pickle(triple_ranks_test_pp1).to_csv('test.tsv')
    #
    # ###
    # # TEST
    # ###

    up_test_data = 'data/TEST/test_TEST.tsv'
    test_data = 'data/pp_twitter_data/TEST/pp_test_TEST.tsv'
    output = 'data/output/output_2a.tsv'





    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio'])
    # featureset_test = fsg.generate_feature_set(test_data)

    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(complete_feature_set_pairs_train, complete_feature_set_pairs_test_TEST, test_data, classification_output)

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(test_data, output, n=5)

    # output = 'data/output/subtask2A_english.tsv'
    #
    # test_data_labels = 'data/TEST/CT2022-Task2A-EN-Test_Qrels_gold.tsv'

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'infersent', 'sim_cse', 'words'], 'TEST')
    # ufsg.create_top_n_output_file(test_data, output, n=5)

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'sim_cse', 'words'], 'TEST')
    # ufsg.create_top_n_output_file(test_data, output, n=5)

    # evaluate_CLEF(test_data_labels, output)









