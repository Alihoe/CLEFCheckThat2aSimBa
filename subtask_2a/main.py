import pandas as pd

from src.evaluation.scorer.main import evaluate_CLEF
from src.feature_generation import pp_old_test_data, old_test_data, complete_feature_set_pairs_test, \
    complete_feature_set_pairs_train, old_predictions_binary, old_test_data_labels, pp_training_data, \
    all_training_data_labels
from src.feature_generation.feature_set_generator import FeatureSetGenerator
from src.feature_generation.file_paths.TEST_file_names import complete_feature_set_pairs_test_TEST
from src.feature_generation.unsupervised_feature_set_generator import UnsupervisedFeatureSetGenerator
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
top_5_sbert_universal_sim_cse_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_features.tsv'
top_5_sbert_universal_sim_cse_ne_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_features.tsv'
top_5_sbert_universal_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_jacc_tok.tsv'
top_5_all_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_all_sentence_embeddings.tsv'
top_5_sbert_universal_sim_cse_ne_ne_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_ne_ratio_features.tsv'
top_5_sbert_universal_sim_cse_ne_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_main_syms_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_main_syms_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words_token_ratio = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words_token_ratio_features.tsv'
top_5_sbert_universal_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_words.tsv'

if __name__ == '__main__':


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

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse', 'words'], 'pp1')
    # ufsg.create_top_n_output_file(old_test_data, top_5_sbert_universal_sim_cse_words, n=5)
    # evaluate_CLEF(old_test_data_labels, top_5_sbert_universal_sim_cse_words) #0.9054

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
    # evaluate_CLEF(old_test_data_labels, top_5_sbert) # 0.8860

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

    classification_output = 'data/predictions/TEST/subtask2A_english.tsv'


    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number', 'ne_ne_ratio',
    #                            'ne_token_ratio', 'main_syms_ratio', 'main_syms_token_ratio', 'words_ratio',
    #                            'words_token_ratio'])
    # featureset_test = fsg.generate_feature_set(test_data)

    predictor = Predictor('binary_classification')
    predictor.train_and_predict(complete_feature_set_pairs_train, complete_feature_set_pairs_test_TEST, test_data, classification_output)

    # ufsg = UnsupervisedFeatureSetGenerator(['sbert', 'universal', 'sim_cse'], 'TEST')
    # ufsg.create_top_n_output_file(test_data, output, n=5)







