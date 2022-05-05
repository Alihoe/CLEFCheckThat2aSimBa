import pandas as pd

from src.feature_generation import v_claims_directory, complete_feature_set_pairs_train, \
    complete_feature_set_triples_train, v_claims_df, complete_feature_set_pairs_test, complete_feature_set_triples_test
from src.feature_generation.feature_set_combiner import FeatureSetCombiner
from src.feature_generation.file_paths.TEST_file_names import complete_feature_set_pairs_train_TEST, \
    complete_feature_set_triples_train_TEST, complete_feature_set_pairs_test_TEST, \
    complete_feature_set_triples_test_TEST
from src.feature_generation.file_paths.pp2_files import complete_feature_set_triples_train_pp2, \
    complete_feature_set_pairs_train_pp2, complete_feature_set_pairs_test_pp2, complete_feature_set_triples_test_pp2
from src.feature_generation.pair_similarity_feature_generator import PairSimilarityFeatureGenerator
from src.feature_generation.sentence_feature_generator import SentenceFeatureGenerator
from src.feature_generation.src.creating_datafiles.data_formatter import DataFormatter
from src.feature_generation.src.creating_datafiles.feature_set_maker import FeatureSetMaker

n = 50


class FeatureSetGenerator:

    def __init__(self, list_of_features):
        self.features = list_of_features

    def create_features(self, dataset):
        # SentenceFeatureGenerator.create_sentence_features(self.features, dataset)
        # PairSimilarityFeatureGenerator.create_pair_similarity_features(self.features, dataset)
        PairSimilarityFeatureGenerator.compute_top_n_sentence_embeddings_features(dataset, n)

    def combine_features(self, dataset, labels=0):
        FeatureSetCombiner.combine_top_50_sentence_embeddings_features(dataset)
        FeatureSetCombiner.add_other_features_to_embedding_features(dataset, self.features)
        FeatureSetCombiner.add_scores_to_feature_set(dataset, labels)

    def prepare_vclaims(self, vclaims):
        SentenceFeatureGenerator.create_sentence_features(self.features, vclaims)
        DataFormatter.ver_claim_directory_to_dataframe(v_claims_directory, v_claims_df)

    def transform_to_triple_dataset(self, data_set):
        if 'train' in data_set or 'dev' in data_set:
            if 'pp2' in data_set:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_training(complete_feature_set_pairs_train_pp2, complete_feature_set_triples_train_pp2)
                df = pd.read_pickle(complete_feature_set_triples_train_pp2 + '.pkl')
            elif 'TEST' in data_set:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_training(complete_feature_set_pairs_train_TEST, complete_feature_set_triples_train_TEST)
                df = pd.read_pickle(complete_feature_set_triples_train_TEST + '.pkl')
            else:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_training(complete_feature_set_pairs_train, complete_feature_set_triples_train)
                df = pd.read_pickle(complete_feature_set_triples_train+'.pkl')
        elif 'test' in data_set:
            if 'pp2' in data_set:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_without_target(complete_feature_set_pairs_test_pp2, complete_feature_set_triples_test_pp2)
                df = pd.read_pickle(complete_feature_set_triples_test_pp2 + '.pkl')
            elif 'TEST' in data_set:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_without_target(complete_feature_set_pairs_test_TEST, complete_feature_set_triples_test_TEST)
                df = pd.read_pickle(complete_feature_set_triples_test_TEST + '.pkl')
            else:
                FeatureSetMaker.transform_dataset_to_dataset_for_triple_classification_without_target(complete_feature_set_pairs_test, complete_feature_set_triples_test)
                df = pd.read_pickle(complete_feature_set_triples_test+'.pkl')
        return df

    def generate_feature_set(self, dataset, labels=0):
        self.create_features(dataset)
        self.combine_features(dataset, labels)
        return self.transform_to_triple_dataset(dataset)

    @staticmethod
    def combine_labels(labels_1, labels_2, combined):
        labels_1_df = pd.read_csv(labels_1, sep='\t',
                                         names=['tweet_id', 'Q0', 'ver_claim_id', '1'], dtype=str)
        labels_2_df = pd.read_csv(labels_2, sep='\t',
                                         names=['tweet_id', 'Q0', 'ver_claim_id', '1'], dtype=str)
        return pd.concat([labels_1_df, labels_2_df]).to_pickle(combined)
