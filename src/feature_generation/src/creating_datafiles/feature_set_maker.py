import pandas as pd


class FeatureSetMaker:

    @staticmethod
    def name_features_pairwise_original_oder(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['id', 'sbert', 'infersent', 'universal', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'cse', 'trans', 'score']
        df.to_pickle(data_frame)

    @staticmethod
    def name_features_original_oder(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['i_claim', 'ver_claim', 'sbert', 'infersent', 'universal', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'cse', 'trans', 'score']
        df.to_pickle(data_frame)

    @staticmethod
    def name_features_pairwise_embeddings_first(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['id', 'sbert', 'infersent', 'universal', 'cse', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'score']
        df.to_pickle(data_frame)

    @staticmethod
    def name_features_embeddings_first(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['i_claim', 'ver_claim', 'sbert', 'infersent', 'universal', 'cse', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'score']
        df.to_pickle(data_frame)


    @staticmethod
    def name_test_features_pairwise_original_oder(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['id', 'sbert', 'infersent', 'universal', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'cse', 'trans']
        df.to_pickle(data_frame)

    @staticmethod
    def name_test_features_original_oder(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['i_claim', 'ver_claim', 'sbert', 'infersent', 'universal', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'cse', 'trans']
        df.to_pickle(data_frame)

    @staticmethod
    def name_test_features_pairwise_embeddings_first(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['id', 'sbert', 'infersent', 'universal', 'cse', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects']
        df.to_pickle(data_frame)

    @staticmethod
    def name_test_features_embeddings_first(data_frame):
        df = pd.read_pickle(data_frame)
        df.columns = ['i_claim', 'ver_claim', 'sbert', 'infersent', 'universal', 'cse', 'sequence_matcher', 'levenshtein','jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects']
        df.to_pickle(data_frame)

    @staticmethod
    def combine_feature_dataframes_conjunction(underlying_data, file_1, file_2, output_df_name):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['i_claim_id', 'iclaim'], dtype=str)
        features_1_df = pd.read_pickle(file_1)
        features_2_df = pd.read_pickle(file_2)
        output_df = pd.DataFrame()
        for row in underlying_df.iterrows():
            i_claim_id = row[1][0]
            this_i_claim_subset_1 = features_1_df[features_1_df['i_claim_id'] == i_claim_id]
            this_i_claim_subset_2 = features_2_df[features_2_df['i_claim_id'] == i_claim_id]
            this_i_claim_df = pd.merge(this_i_claim_subset_1, this_i_claim_subset_2, on=['i_claim_id', 'ver_claim_id'], how='inner')
            output_df = pd.concat([output_df, this_i_claim_df], ignore_index=True)
        output_df.to_pickle(output_df_name)

    @staticmethod
    def combine_feature_dataframes_disjunction(underlying_data, file_1, file_1_extended, file_2, file_2_extended, output_df_name):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['i_claim_id', 'iclaim'], dtype=str)
        features_1_df = pd.read_pickle(file_1)
        features_2_df = pd.read_pickle(file_2)
        file_1_extended_df = pd.read_pickle(file_1_extended)
        file_2_extended_df = pd.read_pickle(file_2_extended)
        output_df = pd.DataFrame()
        for row in underlying_df.iterrows():
            i_claim_id = row[1][0]
            this_i_claim_subset_1 = features_1_df[features_1_df['i_claim_id'] == i_claim_id]
            this_i_claim_subset_2 = features_2_df[features_2_df['i_claim_id'] == i_claim_id]
            this_i_claim_df = pd.merge(this_i_claim_subset_1, this_i_claim_subset_2, on=['i_claim_id', 'ver_claim_id'], how="outer")
            output_df = pd.concat([output_df, this_i_claim_df], ignore_index=True)
        sim_score_x = output_df['sim_score_x']
        output_df = output_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_1_extended_df = file_1_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_2_extended_df = file_2_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score_y.fillna(file_2_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        output_df['sim_score_x'] = sim_score_x
        sim_score_y = output_df['sim_score_y']
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score_x.fillna(file_1_extended_df.set_index(['i_claim_id','ver_claim_id']).sim_score).reset_index()
        output_df['sim_score_y'] = sim_score_y
        output_df.to_pickle(output_df_name)

    @staticmethod
    def add_features_to_dataset(input, features_to_add, output):
        input_df = pd.read_pickle(input)
        input_df = input_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first')
        features_to_add_df = pd.read_pickle(features_to_add)
        features_to_add_df = features_to_add_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first')
        output_df = pd.merge(input_df, features_to_add_df, on=['i_claim_id', 'ver_claim_id'], how="left")
        output_df.to_pickle(output)

    @staticmethod
    def combine_three_feature_dataframes_disjunction(underlying_data, combined_file_1_2, file_1_extended, file_2_extended, file_3, file_3_extended, output_df_name):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['i_claim_id', 'iclaim'], dtype=str)
        features_1_2_df = pd.read_pickle(combined_file_1_2)
        features_3_df = pd.read_pickle(file_3)
        file_1_extended_df = pd.read_pickle(file_1_extended)
        file_2_extended_df = pd.read_pickle(file_2_extended)
        file_3_extended_df = pd.read_pickle(file_3_extended)
        output_df = pd.DataFrame()
        for row in underlying_df.iterrows():
            i_claim_id = row[1][0]
            this_i_claim_subset_1 = features_1_2_df[features_1_2_df['i_claim_id'] == i_claim_id]
            this_i_claim_subset_3 = features_3_df[features_3_df['i_claim_id'] == i_claim_id]
            this_i_claim_df = pd.merge(this_i_claim_subset_1, this_i_claim_subset_3, on=['i_claim_id', 'ver_claim_id'], how="outer")
            output_df = pd.concat([output_df, this_i_claim_df], ignore_index=True)
        sim_score_x = output_df['sim_score_x']
        sim_score_y = output_df['sim_score_y']
        output_df = output_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_1_extended_df = file_1_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_2_extended_df = file_2_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_3_extended_df = file_3_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score.fillna(file_3_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        output_df['sim_score_x'] = sim_score_x
        output_df['sim_score_y'] = sim_score_y
        sim_score = output_df['sim_score']
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score_y.fillna(file_2_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        output_df['sim_score_x'] = sim_score_x
        output_df['sim_score'] = sim_score
        sim_score_y = output_df['sim_score_y']
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score_x.fillna(file_1_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        output_df['sim_score_y'] = sim_score_y
        output_df['sim_score'] = sim_score
        output_df.to_pickle(output_df_name)


    @staticmethod
    def combine_four_feature_dataframes_disjunction(underlying_data, combined_file_1_2_3, file_1_extended,
                                                     file_2_extended, file_3_extended, file_4, file_4_extended, output_df_name):
        underlying_df = pd.read_csv(underlying_data, sep='\t', names=['i_claim_id', 'iclaim'], dtype=str)
        features_1_2_3_df = pd.read_pickle(combined_file_1_2_3)
        features_4_df = pd.read_pickle(file_4)
        file_1_extended_df = pd.read_pickle(file_1_extended)
        file_2_extended_df = pd.read_pickle(file_2_extended)
        file_3_extended_df = pd.read_pickle(file_3_extended)
        file_4_extended_df = pd.read_pickle(file_4_extended)
        output_df = pd.DataFrame()
        for row in underlying_df.iterrows():
            i_claim_id = row[1][0]
            this_i_claim_subset_1 = features_1_2_3_df[features_1_2_3_df['i_claim_id'] == i_claim_id]
            this_i_claim_subset_4 = features_4_df[features_4_df['i_claim_id'] == i_claim_id]
            this_i_claim_df = pd.merge(this_i_claim_subset_1, this_i_claim_subset_4,
                                       on=['i_claim_id', 'ver_claim_id'], how="outer")
            output_df = pd.concat([output_df, this_i_claim_df], ignore_index=True)
        print(output_df.columns)
        output_df.columns = ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse']
        sbert = output_df['sbert']
        infersent = output_df['infersent']
        universal = output_df['universal']
        output_df = output_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_1_extended_df = file_1_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_2_extended_df = file_2_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_3_extended_df = file_3_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        file_4_extended_df = file_4_extended_df.drop_duplicates(subset=['i_claim_id', 'ver_claim_id'], keep='first', ignore_index=True)
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sim_cse.fillna(file_4_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        sim_cse = output_df['sim_cse']
        output_df['sbert'] = sbert
        output_df['infersent'] = infersent
        output_df['universal'] = universal
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).universal.fillna(file_3_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        universal = output_df['universal']
        output_df['sbert'] = sbert
        output_df['infersent'] = infersent
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).infersent.fillna(file_2_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        infersent = output_df['infersent']
        output_df['sbert'] = sbert
        output_df = output_df.set_index(['i_claim_id', 'ver_claim_id']).sbert.fillna(file_1_extended_df.set_index(['i_claim_id', 'ver_claim_id']).sim_score).reset_index()
        output_df['infersent'] = infersent
        output_df['universal'] = universal
        output_df['sim_cse'] = sim_cse
        output_df.to_pickle(output_df_name)

    @staticmethod
    def combine_two_correct_pairs_datasets(correct_i_claim_ver_claim_pairs_1, correct_i_claim_ver_claim_pairs_2, combined):
        correct_pairs_1_df = pd.read_csv(correct_i_claim_ver_claim_pairs_1, sep='\t',
                                         names=['tweet_id', 'Q0', 'ver_claim_id', '1'], dtype=str)
        correct_pairs_2_df = pd.read_csv(correct_i_claim_ver_claim_pairs_2, sep='\t',
                                         names=['tweet_id', 'Q0', 'ver_claim_id', '1'], dtype=str)
        return pd.concat([correct_pairs_1_df, correct_pairs_2_df]).to_pickle(combined)

    @staticmethod
    def get_correct_score_for_tweet_ver_claim_pair(i_claim_id, ver_claim_id, correct_pairs):
        if ((correct_pairs['tweet_id'] == i_claim_id) & (correct_pairs['ver_claim_id'] == ver_claim_id)).any():
            return 1
        else:
            return 0

    @staticmethod
    def add_correct_score_to_dataframe(feature_set, correct_score_dataset, feature_set_with_correct_rankings):
        feature_set_df = pd.read_pickle(feature_set)
        if correct_score_dataset.endswith('.tsv'):
            correct_score_df = pd.read_csv(correct_score_dataset, sep='\t', names=['i_claim_id', 'Q0', 'ver_claim_id', '1'], dtype=str)
        else:
            correct_score_df = pd.read_pickle(correct_score_dataset)
        list_of_scores = []
        old_iclaim_id = ''
        for row in feature_set_df.iterrows():
            i_claim_id = row[1][0]
            if old_iclaim_id == i_claim_id:
                score == '0'
            else:
                ver_claim_id = row[1][1]
                score = FeatureSetMaker.get_correct_score_for_tweet_ver_claim_pair(i_claim_id, ver_claim_id, correct_score_df)
                if score == '1':
                    old_iclaim_id = i_claim_id
            list_of_scores.append(score)
        feature_set_df['score'] = list_of_scores
        feature_set_df.to_pickle(feature_set_with_correct_rankings)
        return feature_set_df

    @staticmethod
    def create_new_id(i_claim_id, ver_claim_id_1, ver_claim_id_2):
        return i_claim_id + '_' + ver_claim_id_1 + '_' + ver_claim_id_2

    @staticmethod
    def transform_dataset_to_dataset_for_triple_classification_training(feature_set_file_path, pairwise_feature_set_file_path):
        input_feature_set = pd.read_pickle(feature_set_file_path)
        col = input_feature_set.columns
        for row in input_feature_set.iterrows():
            relevant_tweet_id = row[1][0]
            relevant_ver_claim_id = row[1][1]
            relevant_score = row[1][len(col) - 1]
            relevant_features_dic = {}
            for i in range(2, len(col) - 1):
                relevant_features_dic[str(i)] = row[1][i]
            subset_criteria = (input_feature_set['i_claim_id'] == relevant_tweet_id) & (input_feature_set['ver_claim_id'] != relevant_ver_claim_id)
            all_other_ver_claims_for_that_tweet_sub_dataset = input_feature_set[subset_criteria]
            for subset_row in all_other_ver_claims_for_that_tweet_sub_dataset.iterrows():
                current_ver_claim_id = subset_row[1][1]
                current_score = subset_row[1][len(col) - 1]
                current_features_dic = {}
                for i in range(2, len(col) - 1):
                    current_features_dic[str(i)] = subset_row[1][i]
                id = FeatureSetMaker.create_new_id(relevant_tweet_id, relevant_ver_claim_id, current_ver_claim_id)
                feature_dist_dic = {}
                for i in range(2, len(col) - 1):
                    feature_dist_dic[str(i)] = str(float(relevant_features_dic[str(i)]) - float(current_features_dic[str(i)]))
                rank_dist = str(float(relevant_score) - float(current_score))
                with open(pairwise_feature_set_file_path+'.tsv', 'a') as output_file:
                    unjoined_list = [id]
                    for key in feature_dist_dic:
                        unjoined_list.append(feature_dist_dic[key])
                    unjoined_list.append(rank_dist)
                    joined_list = "\t".join(unjoined_list)
                    print(joined_list)
                    print(joined_list, file=output_file)
        names = ['id']
        for i in range(2, len(col) - 1):
            names.append('feature_'+str(i-1)+'_dist')
        names.append('rank_dist')
        df = pd.read_csv(pairwise_feature_set_file_path + '.tsv', sep='\t', header=None, names=names)
        df.to_pickle(pairwise_feature_set_file_path + '.pkl')

    @staticmethod
    def transform_dataset_to_dataset_for_triple_classification_without_target(feature_set_file_path, triple_feature_set_file_path):
        input_feature_set = pd.read_pickle(feature_set_file_path)
        col = input_feature_set.columns
        for row in input_feature_set.iterrows():
            relevant_tweet_id = row[1][0]
            relevant_ver_claim_id = row[1][1]
            relevant_features_dic = {}
            for i in range(2, len(col)):
                relevant_features_dic[str(i)] = row[1][i]
            subset_criteria = (input_feature_set['i_claim_id'] == relevant_tweet_id) & (input_feature_set['ver_claim_id'] != relevant_ver_claim_id)
            all_other_ver_claims_for_that_tweet_sub_dataset = input_feature_set[subset_criteria]
            for subset_row in all_other_ver_claims_for_that_tweet_sub_dataset.iterrows():
                current_ver_claim_id = subset_row[1][1]
                current_features_dic = {}
                for i in range(2, len(col)):
                    current_features_dic[str(i)] = subset_row[1][i]
                id = FeatureSetMaker.create_new_id(relevant_tweet_id, relevant_ver_claim_id, current_ver_claim_id)
                feature_dist_dic = {}
                for i in range(2, len(col)):
                    feature_dist_dic[str(i)] = str(float(relevant_features_dic[str(i)]) - float(current_features_dic[str(i)]))
                with open(triple_feature_set_file_path+'.tsv', 'a') as output_file:
                    unjoined_list = [id]
                    for key in feature_dist_dic:
                        unjoined_list.append(feature_dist_dic[key])
                    joined_list = "\t".join(unjoined_list)
                    print(joined_list, file=output_file)
        names = ['id']
        for i in range(2, len(col)):
            names.append('feature_'+str(i-1)+'_dist')
        df = pd.read_csv(triple_feature_set_file_path + '.tsv', sep='\t', header=None, names=names)
        df.to_pickle(triple_feature_set_file_path+'.pkl')




