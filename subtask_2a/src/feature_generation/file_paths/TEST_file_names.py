v_claims_directory_TEST = 'data/politifact-vclaims'
v_claims_df_TEST = 'data/vclaims_df.pkl'

# Complete Feature Sets

# Training

complete_feature_set_pairs_train_TEST = 'data/feature_sets/training/TEST/complete_feature_set_pairs_train_TEST.pkl'
complete_feature_set_pairs_train_tsv_TEST = 'data/feature_sets/training/TEST/complete_feature_set_pairs_train_TEST.tsv'
complete_feature_set_triples_train_TEST = 'data/feature_sets/training/TEST/complete_feature_set_triples_train_TEST'
complete_feature_set_triples_train_tsv_TEST = 'data/feature_sets/training/TEST/complete_feature_set_triples_train_TEST.tsv'

# Test

complete_feature_set_pairs_test_TEST = 'data/feature_sets/test/TEST/complete_feature_set_pairs_test_TEST.pkl'
complete_feature_set_pairs_test_tsv_TEST = 'data/feature_sets/test/TEST/complete_feature_set_pairs_test_TEST.tsv'
complete_feature_set_triples_test_TEST = 'data/feature_sets/test/TEST/complete_feature_set_triples_test_TEST'
complete_feature_set_triples_test_tsv_TEST = 'data/feature_sets/test/TEST/complete_feature_set_triples_test_TEST.tsv'

# Feature Sets for double classification

triple_ranks_training_TEST = 'data/feature_sets/training/TEST/double_classification/triple_ranks_training_TEST.pkl'
triple_ranks_training_labeled_TEST ='data/feature_sets/training/TEST/double_classification/triple_ranks_training_labeled_TEST.pkl'
triple_ranks_test_TEST = 'data/feature_sets/test/TEST/double_classification/triple_ranks_test_TEST.pkl'


### Sentence Features ###

# Verified Claims

# Feature 1
sbert_encodings_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/sbert_encodings_vclaims_TEST.pkl'
sbert_encodings_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/sbert_encodings_vclaims_TEST.tsv'
# Feature 2
infersent_encodings_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/infersent_encodings_vclaims_TEST.pkl'
infersent_encodings_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/infersent_encodings_vclaims_TEST.tsv'
# Feature 3
universal_encodings_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/universal_encodings_vclaims_TEST.pkl'
universal_encodings_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/universal_encodings_vclaims_TEST.tsv'
# Feature 4
sim_cse_encodings_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/sim_cse_encodings_vclaims_TEST.pkl'
sim_cse_encodings_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/sim_cse_encodings_vclaims_TEST.tsv'
# Feature 9
ne_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/ne_vclaims_TEST.pkl'
ne_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/ne_vclaims_TEST.tsv'
# Feature 10
main_syms_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/main_syms_tvclaims_TEST.pkl'
main_syms_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/main_syms_vclaims_TEST.tsv'
# Feature 11
words_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/words_vclaims_TEST.pkl'
words_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/words_tvclaims_TEST.tsv'
# Feature 12
subjects_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/subjects_vclaims_TEST.pkl'
subjects_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/subjects_vclaims_TEST.tsv'
# Feature 13
token_number_vclaims_TEST = 'data/feature_sets/sentence_features/vclaims/TEST/token_number_vclaims_TEST.pkl'
token_number_vclaims_TEST_tsv = 'data/feature_sets/sentence_features/vclaims/TEST/token_number_vclaims_TEST.tsv'

# Training

# Feature 1
sbert_encodings_training_TEST = 'data/feature_sets/sentence_features/training/TEST/sbert_encodings_train_TEST.pkl'
sbert_encodings_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/sbert_encodings_train_TEST.tsv'
# Feature 2
infersent_encodings_training_TEST = 'data/feature_sets/sentence_features/training/TEST/infersent_encodings_train_TEST.pkl'
infersent_encodings_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/infersent_encodings_train_TEST.tsv'
# Feature 3
universal_encodings_training_TEST = 'data/feature_sets/sentence_features/training/TEST/universal_encodings_train_TEST.pkl'
universal_encodings_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/universal_encodings_train_TEST.tsv'
# Feature 4
sim_cse_encodings_training_TEST = 'data/feature_sets/sentence_features/training/TEST/sim_cse_encodings_train_TEST.pkl'
sim_cse_encodings_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/sim_cse_encodings_train_TEST.tsv'
# Feature 9
ne_training_TEST = 'data/feature_sets/sentence_features/training/TEST/ne_train_TEST.pkl'
ne_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/ne_train_TEST.tsv'
# Feature 10
main_syms_training_TEST = 'data/feature_sets/sentence_features/training/TEST/main_syms_train_TEST.pkl'
main_syms_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/main_syms_train_TEST.tsv'
# Feature 11
words_training_TEST = 'data/feature_sets/sentence_features/training/TEST/words_train_TEST.pkl'
words_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/words_train_TEST.tsv'
# Feature 12
subjects_training_TEST = 'data/feature_sets/sentence_features/training/TEST/subjects_train_TEST.pkl'
subjects_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/subjects_train_TEST.tsv'
# Feature 13
token_number_training_TEST = 'data/feature_sets/sentence_features/training/TEST/token_number_train_TEST.pkl'
token_number_training_TEST_tsv = 'data/feature_sets/sentence_features/training/TEST/token_number_train_TEST.tsv'

# Test

# Feature 1
sbert_encodings_test_TEST = 'data/feature_sets/sentence_features/test/TEST/sbert_encodings_test_TEST.pkl'
sbert_encodings_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/sbert_encodings_test_TEST.tsv'
# Feature 2
infersent_encodings_test_TEST = 'data/feature_sets/sentence_features/test/TEST/infersent_encodings_test_TEST.pkl'
infersent_encodings_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/infersent_encodings_test_TEST.tsv'
# Feature 3
universal_encodings_test_TEST = 'data/feature_sets/sentence_features/test/TEST/universal_encodings_test_TEST.pkl'
universal_encodings_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/universal_encodings_test_TEST.tsv'
# Feature 4
sim_cse_encodings_test_TEST = 'data/feature_sets/sentence_features/test/TEST/sim_cse_encodings_test_TEST.pkl'
sim_cse_encodings_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/sim_cse_encodings_test_TEST.tsv'
# Feature 9
ne_test_TEST = 'data/feature_sets/sentence_features/test/TEST/ne_test_TEST.pkl'
ne_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/ne_test_TEST.tsv'
# Feature 10
main_syms_test_TEST = 'data/feature_sets/sentence_features/test/TEST/main_syms_test_TEST.pkl'
main_syms_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/main_syms_test_TEST.tsv'
# Feature 11
words_test_TEST = 'data/feature_sets/sentence_features/test/TEST/words_test_TEST.pkl'
words_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/words_test_TEST.tsv'
# Feature 12
subjects_test_TEST = 'data/feature_sets/sentence_features/test/TEST/subjects_test_TEST.pkl'
subjects_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/subjects_test_TEST.tsv'
# Feature 13
token_number_test_TEST = 'data/feature_sets/sentence_features/test/TEST/token_number_test_TEST.pkl'
token_number_test_TEST_tsv = 'data/feature_sets/sentence_features/test/TEST/token_number_test_TEST.tsv'

### Sentence Similarities ###

# Training

# Feature 1
sbert_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/sbert_sims_train_TEST.pkl'
sbert_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/sbert_sims_train_TEST.tsv'
top_n_sbert_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/top_n_sbert_sims_train_TEST'
top_50_sbert_sims_training_TEST_df = top_n_sbert_sims_training_TEST+'_50.pkl'
top_50_sbert_sims_training_TEST_tsv = top_n_sbert_sims_training_TEST+'_50.tsv'
# Feature 2
infersent_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/infersent_sims_train_TEST.pkl'
infersent_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/infersent_sims_train_TEST.tsv'
top_n_infersent_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/top_n_infersent_sims_train_TEST'
top_50_infersent_sims_training_TEST_df = top_n_infersent_sims_training_TEST+'_50.pkl'
top_50_infersent_sims_training_TEST_tsv = top_n_infersent_sims_training_TEST+'_50.tsv'
# Feature 3
universal_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/universal_sims_train_TEST.pkl'
universal_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/universal_sims_train_TEST.tsv'
top_n_universal_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/top_n_universal_sims_train_TEST'
top_50_universal_sims_training_TEST_df = top_n_universal_sims_training_TEST+'_50.pkl'
top_50_universal_sims_training_TEST_tsv = top_n_universal_sims_training_TEST+'_50.tsv'
# Feature 4
sim_cse_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/sim_cse_sims_train_TEST.pkl'
sim_cse_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/sim_cse_sims_train_TEST.tsv'
top_n_sim_cse_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/top_n_sim_cse_sims_train_TEST'
top_50_sim_cse_sims_training_TEST_df = top_n_sim_cse_sims_training_TEST+'_50.pkl'
top_50_sim_cse_sims_training_TEST_tsv = top_n_sim_cse_sims_training_TEST+'_50.tsv'
# Feature 5
seq_match_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/seq_match_train_TEST.pkl'
seq_match_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/seq_match_train_TEST.tsv'
# Feature 6
levenshtein_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/levenshtein_train_TEST.pkl'
levenshtein_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/levenshtein_train_TEST.tsv'
# Feature 7
jacc_chars_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/jacc_chars_train_TEST.pkl'
jacc_chars_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/jacc_chars_train_TEST.tsv'
# Feature 8
jacc_tokens_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/jacc_tokens_train_TEST.pkl'
jacc_tokens_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/jacc_tokens_train_TEST.tsv'
# Feature 9
ne_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/ne_sims_train_TEST.pkl'
ne_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/ne_sims_train_TEST.tsv'
# Feature 10
main_syms_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_sims_train_TEST.pkl'
main_syms_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_sims_train_TEST.tsv'
# Feature 11
words_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/words_sims_train_TEST.pkl'
words_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/words_sims_train_TEST.tsv'
# Feature 12
subjects_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/subjects_sims_train_TEST.pkl'
subjects_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/subjects_sims_train_TEST.tsv'
# Feature 13
token_number_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/token_number_sims_training_TEST.pkl'
token_number_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/token_number_sims_training_TEST.tsv'
# Feature 14
ne_ne_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/ne_ne_ratio_training_TEST.pkl'
ne_ne_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/ne_ne_ratio_training_TEST.tsv'
# Feature 15
ne_token_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/ne_token_ratio_training_TEST.pkl'
ne_token_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/ne_token_ratio_training_TEST.tsv'
# Feature 16
main_syms_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_ratio_training_TEST.pkl'
main_syms_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_ratio_training_TEST.tsv'
# Feature 17
main_syms_token_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_token_ratio_training_TEST.pkl'
main_syms_token_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/main_syms_token_ratio_training_TEST.tsv'
# Feature 18
words_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/words_ratio_training_TEST.pkl'
words_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/words_ratio_training_TEST.tsv'
# Feature 19
words_token_ratio_sims_training_TEST = 'data/feature_sets/sentence_similarities/training/TEST/words_token_ratio_training_TEST.pkl'
words_token_ratio_sims_training_TEST_tsv = 'data/feature_sets/sentence_similarities/training/TEST/words_token_ratio_training_TEST.tsv'

# Test

# Feature 1
sbert_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/sbert_sims_test_TEST.pkl'
sbert_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/sbert_sims_test_TEST.tsv'
top_n_sbert_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/top_n_sbert_sims_test_TEST'
top_50_sbert_sims_test_TEST_df = top_n_sbert_sims_test_TEST+'_50.pkl'
top_50_sbert_sims_test_TEST_tsv = top_n_sbert_sims_test_TEST+'_50.tsv'
# Feature 2
infersent_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/infersent_sims_test_TEST.pkl'
infersent_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/infersent_sims_test_TEST.tsv'
top_n_infersent_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/top_n_infersent_sims_test_TEST'
top_50_infersent_sims_test_TEST_df = top_n_infersent_sims_test_TEST+'_50.pkl'
top_50_infersent_sims_test_TEST_tsv = top_n_infersent_sims_test_TEST+'_50.tsv'
# Feature 3
universal_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/universal_sims_test_TEST.pkl'
universal_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/universal_sims_test_TEST.tsv'
top_n_universal_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/top_n_universal_sims_test_TEST'
top_50_universal_sims_test_TEST_df = top_n_universal_sims_test_TEST+'_50.pkl'
top_50_universal_sims_test_TEST_tsv = top_n_universal_sims_test_TEST+'_50.tsv'
# Feature 4
sim_cse_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/sim_cse_sims_test_TEST.pkl'
sim_cse_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/sim_cse_sims_test_TEST.tsv'
top_n_sim_cse_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/top_n_sim_cse_sims_test_TEST'
top_50_sim_cse_sims_test_TEST_df = top_n_sim_cse_sims_test_TEST+'_50.pkl'
top_50_sim_cse_sims_test_TEST_tsv = top_n_sim_cse_sims_test_TEST+'_50.tsv'
# Feature 5
seq_match_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/seq_match_test_TEST.pkl'
seq_match_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/seq_match_test_TEST.tsv'
# Feature 6
levenshtein_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/levenshtein_test_TEST.pkl'
levenshtein_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/levenshtein_test_TEST.tsv'
# Feature 7
jacc_chars_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/jacc_chars_test_TEST.pkl'
jacc_chars_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/jacc_chars_test_TEST.tsv'
# Feature 8
jacc_tokens_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/jacc_tokens_test_TEST.pkl'
jacc_tokens_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/jacc_tokens_test_TEST.tsv'
# Feature 9
ne_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/ne_sims_test_TEST.pkl'
ne_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/ne_sims_test_TEST.tsv'
# Feature 10
main_syms_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_sims_test_TEST.pkl'
main_syms_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_sims_test_TEST.tsv'
# Feature 11
words_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/words_sims_test_TEST.pkl'
words_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/words_sims_test_TEST.tsv'
# Feature 12
subjects_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/subjects_sims_test_TEST.pkl'
subjects_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/subjects_sims_test_TEST.tsv'
# Feature 13
token_number_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/token_number_sims_test_TEST.pkl'
token_number_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/token_number_sims_test_TEST.tsv'
# Feature 14
ne_ne_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/ne_ne_ratio_test_TEST.pkl'
ne_ne_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/ne_ne_ratio_test_TEST.tsv'
# Feature 15
ne_token_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/ne_token_ratio_test_TEST.pkl'
ne_token_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/ne_token_ratio_test_TEST.tsv'
# Feature 16
main_syms_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_ratio_test_TEST.pkl'
main_syms_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_ratio_test_TEST.tsv'
# Feature 17
main_syms_token_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_token_ratio_test_TEST.pkl'
main_syms_token_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/main_syms_token_ratio_test_TEST.tsv'
# Feature 18
words_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/words_ratio_test_TEST.pkl'
words_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/words_ratio_test_TEST.tsv'
# Feature 19
words_token_ratio_sims_test_TEST = 'data/feature_sets/sentence_similarities/test/TEST/words_token_ratio_test_TEST.pkl'
words_token_ratio_sims_test_TEST_tsv = 'data/feature_sets/sentence_similarities/test/TEST/words_token_ratio_test_TEST.tsv'

# Combined sentence embedding similarities

## training

train_sbert_infersent_disjunction_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_sbert_infersent_disjunction_TEST.pkl'
train_sbert_infersent_universal_disjunction_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_sbert_infersent_universal_disjunction_TEST.pkl'
train_sbert_infersent_universal_sim_cse_disjunction_TEST = 'data/feature_sets/training/incomplete_feature_sets/TEST/train_sbert_infersent_universal_sim_cse_disjunction_TEST.pkl'
train_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST = 'data/feature_sets/training/incomplete_feature_sets/TEST/train_sbert_infersent_universal_sim_cse_disjunction_TEST.tsv'

## test

test_sbert_infersent_disjunction_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_sbert_infersent_disjunction_TEST.pkl'
test_sbert_infersent_universal_disjunction_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_sbert_infersent_universal_disjunction_TEST.pkl'
test_sbert_infersent_universal_sim_cse_disjunction_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction_TEST.pkl'
test_sbert_infersent_universal_sim_cse_disjunction_tsv_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction_TEST.tsv'

# Combined sentence embeddings similarities + other features

## training

train_first_five_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_five_features_TEST.pkl'
train_first_six_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_six_features_TEST.pkl'
train_first_seven_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_seven_features_TEST.pkl'
train_first_eight_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_eight_features_TEST.pkl'
train_first_nine_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_nine_features_TEST.pkl'
train_first_ten_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_ten_features_TEST.pkl'
train_first_eleven_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_eleven_features_TEST.pkl'
train_first_twelve_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_twelve_features_TEST.pkl'
train_first_thirteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_thirteen_features_TEST.pkl'
train_first_fourteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_fourteen_features_TEST.pkl'
train_first_fifteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_fifteen_features_TEST.pkl'
train_first_sixteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_sixteen_features_TEST.pkl'
train_first_seventeen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_seventeen_features_TEST.pkl'
train_first_eighteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_firsteighteen_features_TEST.pkl'
train_first_nineteen_features_TEST = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_nineteen_features_TEST.pkl'

train_first_nineteen_features_TEST_tsv = 'data/feature_sets/training/TEST/incomplete_feature_sets/train_first_twelve_features_TEST.tsv'

## test

test_first_five_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_five_features_TEST.pkl'
test_first_six_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_six_features_TEST.pkl'
test_first_seven_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_seven_features_TEST.pkl'
test_first_eight_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_eight_features_TEST.pkl'
test_first_nine_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_nine_features_TEST.pkl'
test_first_ten_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_ten_features_TEST.pkl'
test_first_eleven_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_eleven_features_TEST.pkl'
test_first_twelve_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_twelve_features_TEST.pkl'
test_first_thirteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_thirteen_features_TEST.pkl'
test_first_fourteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_fourteen_features_TEST.pkl'
test_first_fifteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_fifteen_features_TEST.pkl'
test_first_sixteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_sixteen_features_TEST.pkl'
test_first_seventeen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_seventeen_features_TEST.pkl'
test_first_eighteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_firsteighteen_features_TEST.pkl'
test_first_nineteen_features_TEST = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_nineteen_features_TEST.pkl'

test_first_nineteen_features_TEST_tsv = 'data/feature_sets/test/TEST/incomplete_feature_sets/test_first_twelve_features_TEST.tsv'

