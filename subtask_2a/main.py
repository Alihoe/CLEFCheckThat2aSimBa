from src.evaluation.scorer.main import evaluate_CLEF
from src.feature_generation import complete_feature_set_triples_train, complete_feature_set_triples_test, \
    complete_feature_set_pairs_train, complete_feature_set_pairs_test, triple_ranks_test_pp1
from src.feature_generation.feature_set_generator import FeatureSetGenerator
from src.feature_generation.file_paths.TEST.TEST_file_names import complete_feature_set_pairs_test_TEST, \
    complete_feature_set_triples_test_TEST
from src.pre_processing import test_data
from src.pre_processing.pre_processor import PreProcessor
from src.prediction.feature_selector import FeatureSelector
from src.prediction.predictor import Predictor
from src.prediction.src.output_formatter import OutputFormatter
import pandas as pd

from src.prediction.unsupervised_ranker import UnsupervisedRanker

training_data = 'data/original_twitter_data/training_data/CT2022-Task2A-EN-Train-Dev_Queries.tsv'
pp_training_data = 'data/pp_twitter_data/training_data/pp_CT2022-Task2A-EN-Train-Dev_Queries.tsv'

training_data_labels_train = 'data/original_twitter_data/training_data/CT2022-Task2A-EN-Train_QRELs.tsv'
training_data_labels_dev = 'data/original_twitter_data/training_data/CT2022-Task2A-EN-Dev_QRELs.tsv'
all_training_data_labels = 'data/original_twitter_data/training_data/all_train.pkl'


old_test_data = 'data/original_twitter_data/test_data/CT2022-Task2A-EN-Dev-Test_Queries.tsv'
pp_old_test_data = 'data/pp_twitter_data/test_data/pp_CT2022-Task2A-EN-Dev-Test_Queries.tsv'
old_test_data_labels = 'data/original_twitter_data/test_data/CT2022-Task2A-EN-Dev-Test_QRELs.tsv'

v_claims = 'data/vclaims'

old_predictions_triple = 'data/predictions/pp1/triple.tsv'
old_predictions_binary_proba = 'data/predictions/pp1/binary_proba.tsv'
old_predictions_binary = 'data/predictions/pp1/binary.tsv'
old_predictions_binary_top_scores = 'data/predictions/pp1/binary_top_scores.tsv'
old_pedictions_highest_se_sims ='data/predictions/pp1/highest_se_sims.tsv'
old_predictions_triple_double_classification = 'data/predictions/pp1/triple_double.tsv'
old_predictions_highest_5_se_sims = 'data/predictions/pp1/predictions_highest_5_se_sims.tsv'
old_predictions_highest_10_se_sims = 'data/predictions/pp1/predictions_highest_10_se_sims.tsv'
old_predictions_highest_50_se_sims = 'data/predictions/pp1/predictions_highest_50_se_sims.tsv'

TEST_data = 'data/TEST/test_TEST.tsv'
pp_TEST_data = 'data/pp_twitter_data/TEST/pp_test_TEST.tsv'
predictions_triple = 'data/predictions/TEST/triple.tsv'
predictions_binary_proba = 'data/predictions/TEST/binary_proba.tsv'
predictions_binary = 'data/predictions/TEST/binary.tsv'
predictions_highest_se_sims ='data/predictions/TEST/highest_se_sims.tsv'
predictions_binary_top_scores = 'data/predictions/TEST/binary_top_scores.tsv'

feature_correlation_training_data_spearman = 'data/evaluation/feature_correlation_training_data_spearman.tsv'


if __name__ == '__main__':


    # FeatureSelector.feature_correlation(complete_feature_set_pairs_train, feature_correlation_training_data_spearman)
    # FeatureSelector.mutual_information_feature_selection(complete_feature_set_pairs_train)

    top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_sim_cse.tsv'
    top_5_sbert = 'data/unsupervised_ranking/pp1/top_5_sbert.tsv'
    top_5_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok.tsv'
    top_5_sim_cse_jacc_tok_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_words.tsv'
    top_5_sim_cse_words = 'data/unsupervised_ranking/pp1/top_5_sim_cse_words.tsv'
    top_5_sim_cse_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_ne.tsv'
    top_5_sim_cse_jacc_tok_ne = 'data/unsupervised_ranking/pp1/top_5_sim_cse_jacc_tok_ne.tsv'
    top_5_all_features = 'data/unsupervised_ranking/pp1/top_5_all_features.tsv'
    top_5_all_features_without_infersent = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    top_5_sim_cse = 'data/unsupervised_ranking/pp1/top_5_all_features_without_infersent.tsv'
    top_5_no_sentence_embeddings = 'data/unsupervised_ranking/pp1/top_5_no_sentence_embeddings.tsv'
    top_5_sbert_universal_sim_cse_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_features.tsv'
    top_5_sbert_universal_sim_cse_ne_features = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_ne_features.tsv'
    top_5_sbert_universal_sim_cse_jacc_tok = 'data/unsupervised_ranking/pp1/top_5_sbert_universal_sim_cse_jacc_tok.tsv'


    # ranker = UnsupervisedRanker(['sbert', 'infersent', 'universal', 'sim_cse', 'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_all_features)
    # evaluate_CLEF(old_test_data_labels,  top_5_all_features)

    # ranker = UnsupervisedRanker(['sbert'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_all_features)
    # evaluate_CLEF(old_test_data_labels,  top_5_all_features) # 0.8860

    # ranker = UnsupervisedRanker(['sbert', 'universal', 'sim_cse'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_sbert_universal_sim_cse_features)
    # evaluate_CLEF(old_test_data_labels,  top_5_sbert_universal_sim_cse_features) # 0.9217

    # ranker = UnsupervisedRanker(['sbert', 'universal', 'sim_cse', 'ne'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_sbert_universal_sim_cse_ne_features)
    # evaluate_CLEF(old_test_data_labels,  top_5_sbert_universal_sim_cse_ne_features) # 0.9142

    # ranker = UnsupervisedRanker(['sim_cse'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_sim_cse)
    # evaluate_CLEF(old_test_data_labels,  top_5_sim_cse) # 0.8015

    # ranker = UnsupervisedRanker(['sbert', 'universal', 'sim_cse', 'jacc_tok'])
    # ranker.create_top_n_output_file(complete_feature_set_pairs_test, old_test_data, top_5_sbert_universal_sim_cse_jacc_tok)
    # evaluate_CLEF(old_test_data_labels,  top_5_sbert_universal_sim_cse_jacc_tok) # 0.9201


    # pre_processor = PreProcessor('cleaning_tweets')
    # pp_training_data = pre_processor.pre_process(training_data, pp_training_data)
    # pp_old_test_data = pre_processor.pre_process(old_test_data, pp_old_test_data)

    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects', 'token_number'])
    # fsg.prepare_vclaims(v_claims)
    #
    # labels = fsg.combine_labels(training_data_labels_train, training_data_labels_dev, all_training_data_labels)
    # labels = all_training_data_labels
    # featureset_train = fsg.generate_feature_set(pp_training_data, labels)
    #
    fsg = FeatureSetGenerator(['ne_ne_ratio'])
    fsg.generate_feature_set(pp_old_test_data)
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
    #
    # predictions_triple = 'data/predictions/TEST/triple.tsv'
    #
    # fsg = FeatureSetGenerator(['sbert', 'infersent', 'universal', 'sim_cse', 'seq_match', 'levenshtein', 'jacc_chars',
    #                            'jacc_tokens', 'ne', 'main_syms', 'words', 'subjects'])
    # pre_processor = PreProcessor('cleaning_tweets')
    # pp_TEST_data = pre_processor.pre_process(TEST_data, pp_TEST_data)
    #
    # featureset_test = fsg.generate_feature_set(pp_TEST_data)
    #
    # featureset_train = complete_feature_set_pairs_train
    # featureset_test = complete_feature_set_pairs_test_TEST
    #
    # predictor = Predictor('binary_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, TEST_data, predictions_binary)
    #
    # predictor = Predictor('highest_se_sims')
    # predictor.train_and_predict(featureset_train, featureset_test, TEST_data, predictions_highest_se_sims)
    #
    # predictor = Predictor('binary_proba')
    # predictor.train_and_predict(featureset_train, featureset_test, TEST_data, predictions_binary_proba)
    #
    #
    # OutputFormatter.drop_all_but_top_ver_claims(predictions_binary, predictions_binary_top_scores)
    #
    # featureset_train = complete_feature_set_triples_train + '.pkl'
    # featureset_test = complete_feature_set_triples_test_TEST + '.pkl'
    #
    # predictor = Predictor('triple_classification')
    # predictor.train_and_predict(featureset_train, featureset_test, TEST_data, predictions_triple)
    #
    #
    #
    #
