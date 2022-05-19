import enum
import pandas as pd

from src.feature_generation import training_data_general, labels_general, triple_ranks_training_pp1, \
    triple_ranks_test_pp1, triple_ranks_training_labeled_pp1
from src.feature_generation.file_paths.TEST_file_names import triple_ranks_training_TEST, triple_ranks_test_TEST, \
    triple_ranks_training_labeled_TEST
from src.feature_generation.src.creating_datafiles.feature_set_maker import FeatureSetMaker
from src.prediction.src.output_formatter import OutputFormatter
from src.prediction.src.rank_classifier import RankClassifier
from src.prediction.src.triple_ranker import TripleRanker


class ClassificationMode(enum.Enum):
    binary_proba = 1
    binary_classification = 2
    triple_classification = 3
    triple_classification_with_rank_classification = 4
    highest_se_sims = 5
    highest_n_se_sims = 6


class Predictor:

    def __init__(self, classification_mode):
        if classification_mode == ClassificationMode.binary_proba.name:
            self.ranker = RankClassifier()
            self.classification_mode = ClassificationMode.binary_proba.name
        elif classification_mode == ClassificationMode.binary_classification.name:
            self.ranker = RankClassifier()
            self.classification_mode = ClassificationMode.binary_classification.name
        elif classification_mode == ClassificationMode.triple_classification.name:
            self.ranker = TripleRanker()
            self.classification_mode = ClassificationMode.triple_classification.name
        elif classification_mode == ClassificationMode.triple_classification_with_rank_classification.name:
            self.ranker = TripleRanker()
            self.classification_mode = ClassificationMode.triple_classification_with_rank_classification.name
        elif classification_mode == ClassificationMode.highest_se_sims.name:
            self.classification_mode = ClassificationMode.highest_se_sims.name
        elif classification_mode == ClassificationMode.highest_n_se_sims.name:
            self.classification_mode = ClassificationMode.highest_n_se_sims.name

    def train_and_predict(self, complete_feature_set, training_feature_set, test_feature_set, test_data, output_data, n=5):
        if isinstance(training_feature_set, str):
            training_df = pd.read_pickle(training_feature_set)
        else:
            training_df = training_feature_set
        if isinstance(test_feature_set, str):
            test_df = pd.read_pickle(test_feature_set)
        else:
            test_df = test_feature_set
        if self.classification_mode == ClassificationMode.highest_se_sims.name:
            OutputFormatter.rank_by_highest_sentence_embeddings_score(test_df, test_data, output_data)
        elif self.classification_mode == ClassificationMode.highest_n_se_sims.name:
            OutputFormatter.rank_by_top_n_sentence_embeddings_score(test_df, test_data, output_data, n)
        elif isinstance(self.ranker, TripleRanker):
            model = self.ranker.training_of_triple_classifier(training_df)
            output = self.ranker.predict_triple_rankings(model, test_feature_set)
            if self.classification_mode == ClassificationMode.triple_classification.name:
                OutputFormatter.retransform_dataset_for_triple_classification_to_ranked_feature_dataset(output, test_data, output_data)
            elif self.classification_mode == ClassificationMode.triple_classification_with_rank_classification.name:
                training_data = training_data_general
                labels = labels_general
                if 'pp1' in training_feature_set:
                    triple_ranks_training = triple_ranks_training_pp1
                    triple_ranks_test = triple_ranks_test_pp1
                    triple_ranks_training_labeled = triple_ranks_training_labeled_pp1
                elif 'TEST' in training_feature_set:
                    triple_ranks_training = triple_ranks_training_TEST
                    triple_ranks_test = triple_ranks_test_TEST
                    triple_ranks_training_labeled = triple_ranks_training_labeled_TEST
                training_feature_set_df = pd.read_pickle(training_feature_set)
                OutputFormatter.transform_scored_dataset_of_triple_classification(training_feature_set_df, training_data,
                                                                                  triple_ranks_training)
                FeatureSetMaker.add_correct_score_to_dataframe(triple_ranks_training, labels, triple_ranks_training_labeled)
                OutputFormatter.transform_scored_dataset_of_triple_classification(output, test_data, triple_ranks_test)
                second_classifier = RankClassifier()
                model = second_classifier.training_of_classifier(triple_ranks_training_labeled)
                output_second_clasifier = second_classifier.predict_rankings(model, triple_ranks_test)
                OutputFormatter.format_double_output(output_second_clasifier, test_data, output_data)
        elif isinstance(self.ranker, RankClassifier):
            if self.classification_mode == ClassificationMode.binary_classification.name:
                model = self.ranker.training_of_classifier(training_df)
                output = self.ranker.predict_rankings(model, test_feature_set)
                OutputFormatter.format_binary_output(complete_feature_set, output, test_data, output_data)
            elif self.classification_mode == ClassificationMode.binary_proba.name:
                model = self.ranker.training_of_prediction_classifier(training_df)
                output = self.ranker.predict_probabilities(model, test_feature_set)
                OutputFormatter.format_probability_output(output, test_data, output_data)

