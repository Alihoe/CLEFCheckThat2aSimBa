import enum
import numpy as np
import pandas as pd
from scipy.spatial import distance

from src.feature_generation.src.computing_similarity.jacquard_distance_string_sim_computer import \
    JacquardDistanceComputer
from src.feature_generation.src.computing_similarity.jacquard_distance_tokenization_string_sim_computer import \
    JacquardDistanceTokenizationComputer
from src.feature_generation.src.computing_similarity.levenshtein_distance_string_sim_computer import \
    LevenshteinDistanceComputer
from src.feature_generation.src.computing_similarity.list_sim_computer import ListEntitySimComputer
from src.feature_generation.src.computing_similarity.sequence_matching_string_sim_computer import \
    SequenceMatchingComputer
from src.feature_generation.src.computing_similarity.syntactic_similarity_computer import SyntacticSimComputer


class SimScoreType(enum.Enum):
    cosine_sim = 1
    sequence_matching_sim = 2
    levenshtein_dist = 3
    jacquard_dist = 4
    jacquard_dist_token = 5
    ne_sim = 6
    syn_sim = 7
    subject_sim = 8
    token_number_sim = 9
    ne_ne_ratio_sim = 10
    ne_token_ratio_sim = 11
    main_syms_ratio = 12
    words_ratio = 13
    words_token_ratio = 14
    main_syms_token_ratio = 15


class EncodingMethod(enum.Enum):
    full_text = 1
    single_sentences = 2


class SimilarityScoreDataGenerator:

    def __init__(self, sim_score_type, encoding_method='full_text'):
        if sim_score_type == SimScoreType.cosine_sim.name:
            self.sim_score_computer = SimScoreType.cosine_sim.name
        elif sim_score_type == SimScoreType.sequence_matching_sim.name:
            self.sim_score_computer = SequenceMatchingComputer
        elif sim_score_type == SimScoreType.levenshtein_dist.name:
            self.sim_score_computer = LevenshteinDistanceComputer
        elif sim_score_type == SimScoreType.jacquard_dist.name:
            self.sim_score_computer = JacquardDistanceComputer
        elif sim_score_type == SimScoreType.jacquard_dist_token.name:
            self.sim_score_computer = JacquardDistanceTokenizationComputer
        elif sim_score_type == SimScoreType.ne_sim.name:
            self.sim_score_computer = SimScoreType.ne_sim.name
        elif sim_score_type == SimScoreType.syn_sim.name:
            self.sim_score_computer = SimScoreType.syn_sim.name
        elif sim_score_type == SimScoreType.subject_sim.name:
            self.sim_score_computer = SimScoreType.subject_sim.name
        elif sim_score_type == SimScoreType.token_number_sim.name:
            self.sim_score_computer = SimScoreType.token_number_sim.name
        elif sim_score_type == SimScoreType.ne_ne_ratio_sim.name:
            self.sim_score_computer = SimScoreType.ne_ne_ratio_sim.name
        elif sim_score_type == SimScoreType.ne_token_ratio_sim.name:
            self.sim_score_computer = SimScoreType.ne_token_ratio_sim.name
        elif sim_score_type == SimScoreType.words_ratio.name:
            self.sim_score_computer = SimScoreType.words_ratio.name
        elif sim_score_type == SimScoreType.main_syms_ratio.name:
            self.sim_score_computer = SimScoreType.main_syms_ratio.name
        elif sim_score_type == SimScoreType.words_token_ratio.name:
            self.sim_score_computer = SimScoreType.words_token_ratio.name
        elif sim_score_type == SimScoreType.main_syms_token_ratio.name:
            self.sim_score_computer = SimScoreType.main_syms_token_ratio.name

    def generate_all_sim_scores(self, input_claims, ver_claims, all_sim_scores_file, token_numbers=0, vclaims_tokens=0):
        if self.sim_score_computer == SimScoreType.cosine_sim.name or \
                self.sim_score_computer == SimScoreType.ne_sim.name or \
                self.sim_score_computer == SimScoreType.syn_sim.name or\
                self.sim_score_computer == SimScoreType.subject_sim.name:
            input_claims_df = pd.read_pickle(input_claims)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claims_col = input_claims_df.columns
            input_claim_ids = input_claims_df[[input_claims_col[0]]].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            if self.sim_score_computer == SimScoreType.cosine_sim.name:
                input_claim_embeddings = input_claims_df[[input_claims_col[1]]].values
                ver_claim_embeddings_matrix = ver_claims_df[[ver_claims_col[1]]].to_numpy()
                ver_claim_embeddings_matrix = np.reshape(ver_claim_embeddings_matrix, (ver_claim_embeddings_matrix.shape[0],))
                ver_claim_embeddings_matrix = np.stack(ver_claim_embeddings_matrix)
                for i in range(len(input_claim_ids)):
                    this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                    claim_id = input_claim_ids[i][0]
                    print(claim_id)
                    claim_embedding = input_claim_embeddings[i]
                    claim_embedding = np.stack(claim_embedding)
                    nrows = ver_claim_embeddings_matrix.shape[0]
                    this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                    this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                    dist = distance.cdist(claim_embedding, ver_claim_embeddings_matrix, "cosine")[0]
                    sim = (1-dist)*100
                    this_i_claim_df['sim_score'] = sim
                    sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
                sim_score_df.to_pickle(all_sim_scores_file)
            elif self.sim_score_computer == SimScoreType.ne_sim.name or\
                    self.sim_score_computer == SimScoreType.syn_sim.name:
                input_claim_entity_lists = input_claims_df[[input_claims_col[1]]].values
                ver_claim_entity_lists_df = ver_claims_df[[ver_claims_col[1]]]
                for i in range(len(input_claim_ids)):
                    this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                    claim_id = input_claim_ids[i][0]
                    print(claim_id)
                    this_claim_entity_list = input_claim_entity_lists[i]
                    nrows = ver_claim_entity_lists_df.shape[0]
                    this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                    this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                    this_i_claim_df['sim_score'] = ver_claim_entity_lists_df.apply(
                        ListEntitySimComputer.comp_similarity, axis=1, args=[this_claim_entity_list])
                    sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
                sim_score_df.to_pickle(all_sim_scores_file)
            else:
                if self.sim_score_computer == SimScoreType.subject_sim.name:
                    input_claim_entity_lists = input_claims_df[[input_claims_col[1]]].values
                    ver_claim_entity_lists_df = ver_claims_df[[ver_claims_col[1]]]
                    for i in range(len(input_claim_ids)):
                        this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                        claim_id = input_claim_ids[i][0]
                        print(claim_id)
                        this_claim_entity_list = input_claim_entity_lists[i]
                        nrows = ver_claim_entity_lists_df.shape[0]
                        this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                        this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                        this_i_claim_df['sim_score'] = ver_claim_entity_lists_df.apply(
                            SyntacticSimComputer.comp_similarity, axis=1, args=[this_claim_entity_list])
                        sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
                    sim_score_df.to_pickle(all_sim_scores_file)
        elif self.sim_score_computer == SequenceMatchingComputer or \
                self.sim_score_computer == LevenshteinDistanceComputer or \
                self.sim_score_computer == JacquardDistanceComputer or \
                self.sim_score_computer == JacquardDistanceTokenizationComputer:
            input_claims_df = pd.read_csv(input_claims, sep='\t', names=['i_claim_id', 'i_claim'], dtype=str)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claim_ids = input_claims_df['i_claim_id'].values
            input_claims_texts = input_claims_df['i_claim'].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            ver_claim_texts = ver_claims_df[[ver_claims_col[1]]]
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            for i in range(len(input_claim_ids)):
                this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                claim_id = input_claim_ids[i]
                print(claim_id)
                i_claim = input_claims_texts[i]
                nrows = len(ver_claim_ids)
                this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                this_i_claim_df['sim_score'] = ver_claim_texts.apply(
                    self.sim_score_computer.comp_similarity, axis=1, args=[i_claim])
                sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
            sim_score_df.to_pickle(all_sim_scores_file)
        elif self.sim_score_computer == SimScoreType.token_number_sim.name:
            input_claims_df = pd.read_pickle(input_claims)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claims_col = input_claims_df.columns
            input_claim_ids = input_claims_df[[input_claims_col[0]]].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            input_claim_token_numbers = input_claims_df[[input_claims_col[1]]].values
            ver_claim_token_numbers = np.array(ver_claims_df[[ver_claims_col[1]]].values)
            nrows = ver_claims_df[[ver_claims_col[1]]].shape[0]
            for i in range(len(input_claim_ids)):
                this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                claim_id = input_claim_ids[i][0]
                this_claim_number_of_tokens = input_claim_token_numbers[i][0]
                this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                this_i_claim_df['sim_score'] = abs(ver_claim_token_numbers-this_claim_number_of_tokens)
                sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
            sim_score_df.to_pickle(all_sim_scores_file)
        elif self.sim_score_computer == SimScoreType.ne_ne_ratio_sim.name or \
                self.sim_score_computer == SimScoreType.main_syms_ratio.name or \
                self.sim_score_computer == SimScoreType.words_ratio.name:
            input_claims_df = pd.read_pickle(input_claims)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claims_col = input_claims_df.columns
            input_claim_ids = input_claims_df[[input_claims_col[0]]].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            input_claim_entity_lists = input_claims_df[[input_claims_col[1]]].values
            ver_claim_entity_lists_df = ver_claims_df[[ver_claims_col[1]]]
            for i in range(len(input_claim_ids)):
                this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                claim_id = input_claim_ids[i][0]
                print(claim_id)
                this_claim_entity_list = input_claim_entity_lists[i]
                nrows = ver_claim_entity_lists_df.shape[0]
                this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                this_i_claim_df['sim_score'] = ver_claim_entity_lists_df.apply(
                    ListEntitySimComputer.comp_ratio, axis=1, args=[this_claim_entity_list])
                sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
            sim_score_df.to_pickle(all_sim_scores_file)
        elif self.sim_score_computer == SimScoreType.ne_token_ratio_sim.name:
            input_claims_df = pd.read_pickle(input_claims)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claims_col = input_claims_df.columns
            input_claim_ids = input_claims_df[[input_claims_col[0]]].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            input_claim_entity_lists = input_claims_df[[input_claims_col[1]]].values
            ver_claim_entity_lists_df = ver_claims_df[[ver_claims_col[1]]]
            input_claim_tokens_df = pd.read_pickle(token_numbers)
            ver_claims_tokens_df = pd.read_pickle(vclaims_tokens)
            input_claims_tokens_col = input_claim_tokens_df.columns
            ver_claims_tokens_col = ver_claims_tokens_df.columns
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            input_claim_token_numbers = input_claim_tokens_df[[input_claims_tokens_col[1]]].values
            ver_claim_token_numbers = ver_claims_tokens_df[[ver_claims_tokens_col[1]]]
            ver_claim_entities_and_tokens = pd.concat([ver_claim_entity_lists_df.reset_index(drop=True), ver_claim_token_numbers], axis=1)
            for i in range(len(input_claim_ids)):
                this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                claim_id = input_claim_ids[i][0]
                print(claim_id)
                this_claim_entity_list = input_claim_entity_lists[i]
                this_claim_token_number = input_claim_token_numbers[i]
                nrows = ver_claim_entity_lists_df.shape[0]
                this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                this_i_claim_df['sim_score'] = ver_claim_entities_and_tokens.apply(
                    ListEntitySimComputer.comp_token_ne_ratio, axis=1, args=[this_claim_entity_list, this_claim_token_number])
                sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
            sim_score_df.to_pickle(all_sim_scores_file)
        elif self.sim_score_computer == SimScoreType.words_token_ratio.name or\
                self.sim_score_computer == SimScoreType.main_syms_token_ratio.name:
            input_claims_df = pd.read_pickle(input_claims)
            ver_claims_df = pd.read_pickle(ver_claims)
            input_claims_col = input_claims_df.columns
            input_claim_ids = input_claims_df[[input_claims_col[0]]].values
            ver_claims_col = ver_claims_df.columns
            ver_claim_ids = ver_claims_df[[ver_claims_col[0]]].values.tolist()
            input_claim_entity_lists = input_claims_df[[input_claims_col[1]]].values
            ver_claim_entity_lists_df = ver_claims_df[[ver_claims_col[1]]]
            input_claim_tokens_df = pd.read_pickle(token_numbers)
            ver_claims_tokens_df = pd.read_pickle(vclaims_tokens)
            input_claims_tokens_col = input_claim_tokens_df.columns
            ver_claims_tokens_col = ver_claims_tokens_df.columns
            sim_score_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
            input_claim_token_numbers = input_claim_tokens_df[[input_claims_tokens_col[1]]].values
            ver_claim_token_numbers = ver_claims_tokens_df[[ver_claims_tokens_col[1]]]
            ver_claim_entities_and_tokens = pd.concat([ver_claim_entity_lists_df.reset_index(drop=True), ver_claim_token_numbers], axis=1)
            for i in range(len(input_claim_ids)):
                this_i_claim_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
                claim_id = input_claim_ids[i][0]
                print(claim_id)
                this_claim_entity_list = input_claim_entity_lists[i]
                this_claim_token_number = input_claim_token_numbers[i]
                nrows = ver_claim_entity_lists_df.shape[0]
                this_i_claim_df['ver_claim_id'] = ListEntitySimComputer.add_flatten_lists(ver_claim_ids)
                this_i_claim_df['i_claim_id'] = pd.Series([claim_id] * nrows)
                this_i_claim_df['sim_score'] = ver_claim_entities_and_tokens.apply(
                    ListEntitySimComputer.comp_token_ratio, axis=1, args=[this_claim_entity_list, this_claim_token_number])
                sim_score_df = pd.concat([sim_score_df, this_i_claim_df])
            sim_score_df.to_pickle(all_sim_scores_file)


    @staticmethod
    def generate_top_n_from_top_all_sim_scores(all_sim_scores_file, n, top_n_sim_scores_file):
        all_sim_scores_df = pd.read_pickle(all_sim_scores_file)
        top_n_sim_scores_file_name = top_n_sim_scores_file + '_' + str(n) + '.pkl'
        groups = all_sim_scores_df.groupby(all_sim_scores_df['i_claim_id'])
        output_df = pd.DataFrame(columns=['i_claim_id', 'ver_claim_id', 'sim_score'])
        for name, group in groups:
            this_i_claim_df = group.sort_values('sim_score', ascending=False)
            this_i_claim_df = this_i_claim_df.head(n)
            output_df = pd.concat([output_df, this_i_claim_df], ignore_index=True)
        output_df.to_pickle(top_n_sim_scores_file_name)
