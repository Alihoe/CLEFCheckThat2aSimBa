import enum


class Features(enum.Enum):
    sbert = 1
    infersent = 2
    universal = 3
    sim_cse = 4
    seq_match = 5
    levenshtein = 6
    jacc_chars = 7
    jacc_tokens = 8
    ne = 9
    main_syms = 10
    words = 11
    subjects = 12
    token_number = 13
    ne_ne_ratio = 14
    ne_token_ratio = 15
    main_syms_token_ratio = 16
    words_token_ratio = 17


v_claims_directory = 'data/vclaims'
v_claims_df = 'data/vclaims_df.pkl'
training_data_general = 'data/original_twitter_data/training_data/CT2022-Task2A-EN-Train-Dev_Queries.tsv'
labels_general = 'data/original_twitter_data/training_data/all_train.pkl'

# Complete Feature Sets

# Training

complete_feature_set_pairs_train = 'data/feature_sets/training/pp1/complete_feature_set_pairs_train_pp1.pkl'
complete_feature_set_pairs_train_tsv = 'data/feature_sets/training/pp1/complete_feature_set_pairs_train_pp1.tsv'
complete_feature_set_triples_train = 'data/feature_sets/training/pp1/complete_feature_set_triples_train_pp1'
complete_feature_set_triples_train_tsv = 'data/feature_sets/training/pp1/complete_feature_set_triples_train_pp1.tsv'

# Test

complete_feature_set_pairs_test = 'data/feature_sets/test/pp1/complete_feature_set_pairs_test_pp1.pkl'
complete_feature_set_pairs_test_tsv = 'data/feature_sets/test/pp1/complete_feature_set_pairs_test_pp1.tsv'
complete_feature_set_triples_test = 'data/feature_sets/test/pp1/complete_feature_set_triples_test_pp1'
complete_feature_set_triples_test_tsv = 'data/feature_sets/test/pp1/complete_feature_set_triples_test_pp1.tsv'


# Feature Sets for double classification

triple_ranks_training_pp1 = 'data/feature_sets/training/pp1/double_classification/triple_ranks_training_pp1.pkl'
triple_ranks_training_labeled_pp1 ='data/feature_sets/training/pp1/double_classification/triple_ranks_training_labeled_pp1.pkl'
triple_ranks_test_pp1 = 'data/feature_sets/test/pp1/double_classification/triple_ranks_test_pp1.pkl'


### Sentence Features ###

# Verified Claims

# Feature 1
sbert_encodings_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/sbert_encodings_vclaims_pp1.pkl'
sbert_encodings_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/sbert_encodings_vclaims_pp1.tsv'
# Feature 2
infersent_encodings_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/infersent_encodings_vclaims_pp1.pkl'
infersent_encodings_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/infersent_encodings_vclaims_pp1.tsv'
# Feature 3
universal_encodings_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/universal_encodings_vclaims_pp1.pkl'
universal_encodings_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/universal_encodings_vclaims_pp1.tsv'
# Feature 4
sim_cse_encodings_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/sim_cse_encodings_vclaims_pp1.pkl'
sim_cse_encodings_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/sim_cse_encodings_vclaims_pp1.tsv'
# Feature 9
ne_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/ne_vclaims_pp1.pkl'
ne_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/ne_vclaims_pp1.tsv'
# Feature 10
main_syms_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/main_syms_tvclaims_pp1.pkl'
main_syms_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/main_syms_vclaims_pp1.tsv'
# Feature 11
words_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/words_vclaims_pp1.pkl'
words_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/words_tvclaims_pp1.tsv'
# Feature 12
subjects_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/subjects_vclaims_pp1.pkl'
subjects_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/subjects_vclaims_pp1.tsv'
# Feature 13
token_number_vclaims_pp1 = 'data/feature_sets/sentence_features/vclaims/pp1/token_number_vclaims_pp1.pkl'
token_number_vclaims_pp1_tsv = 'data/feature_sets/sentence_features/vclaims/pp1/token_number_vclaims_pp1.tsv'


# Training

# Feature 1
sbert_encodings_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/sbert_encodings_train_pp1.pkl'
sbert_encodings_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/sbert_encodings_train_pp1.tsv'
# Feature 2
infersent_encodings_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/infersent_encodings_train_pp1.pkl'
infersent_encodings_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/infersent_encodings_train_pp1.tsv'
# Feature 3
universal_encodings_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/universal_encodings_train_pp1.pkl'
universal_encodings_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/universal_encodings_train_pp1.tsv'
# Feature 4
sim_cse_encodings_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/sim_cse_encodings_train_pp1.pkl'
sim_cse_encodings_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/sim_cse_encodings_train_pp1.tsv'
# Feature 9
ne_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/ne_train_pp1.pkl'
ne_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/ne_train_pp1.tsv'
# Feature 10
main_syms_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/main_syms_train_pp1.pkl'
main_syms_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/main_syms_train_pp1.tsv'
# Feature 11
words_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/words_train_pp1.pkl'
words_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/words_train_pp1.tsv'
# Feature 12
subjects_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/subjects_train_pp1.pkl'
subjects_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/subjects_train_pp1.tsv'
# Feature 13
token_number_training_pp1 = 'data/feature_sets/sentence_features/training/pp1/token_number_train_pp1.pkl'
token_number_training_pp1_tsv = 'data/feature_sets/sentence_features/training/pp1/token_number_train_pp1.tsv'

# Test

# Feature 1
sbert_encodings_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/sbert_encodings_test_pp1.pkl'
sbert_encodings_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/sbert_encodings_test_pp1.tsv'
# Feature 2
infersent_encodings_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/infersent_encodings_test_pp1.pkl'
infersent_encodings_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/infersent_encodings_test_pp1.tsv'
# Feature 3
universal_encodings_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/universal_encodings_test_pp1.pkl'
universal_encodings_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/universal_encodings_test_pp1.tsv'
# Feature 4
sim_cse_encodings_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/sim_cse_encodings_test_pp1.pkl'
sim_cse_encodings_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/sim_cse_encodings_test_pp1.tsv'
# Feature 9
ne_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/ne_test_pp1.pkl'
ne_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/ne_test_pp1.tsv'
# Feature 10
main_syms_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/main_syms_test_pp1.pkl'
main_syms_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/main_syms_test_pp1.tsv'
# Feature 11
words_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/words_test_pp1.pkl'
words_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/words_test_pp1.tsv'
# Feature 12
subjects_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/subjects_test_pp1.pkl'
subjects_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/subjects_test_pp1.tsv'
# Feature 13
token_number_test_pp1 = 'data/feature_sets/sentence_features/test/pp1/token_number_test_pp1.pkl'
token_number_test_pp1_tsv = 'data/feature_sets/sentence_features/test/pp1/token_number_test_pp1.tsv'

### Sentence Similarities ###

# Training

# Feature 1
sbert_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/sbert_sims_train_pp1.pkl'
sbert_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/sbert_sims_train_pp1.tsv'
top_n_sbert_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/top_n_sbert_sims_train_pp1'
top_50_sbert_sims_training_pp1_df = top_n_sbert_sims_training_pp1+'_50.pkl'
top_50_sbert_sims_training_pp1_tsv = top_n_sbert_sims_training_pp1+'_50.tsv'
# Feature 2
infersent_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/infersent_sims_train_pp1.pkl'
infersent_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/infersent_sims_train_pp1.tsv'
top_n_infersent_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/top_n_infersent_sims_train_pp1'
top_50_infersent_sims_training_pp1_df = top_n_infersent_sims_training_pp1+'_50.pkl'
top_50_infersent_sims_training_pp1_tsv = top_n_infersent_sims_training_pp1+'_50.tsv'
# Feature 3
universal_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/universal_sims_train_pp1.pkl'
universal_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/universal_sims_train_pp1.tsv'
top_n_universal_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/top_n_universal_sims_train_pp1'
top_50_universal_sims_training_pp1_df = top_n_universal_sims_training_pp1+'_50.pkl'
top_50_universal_sims_training_pp1_tsv = top_n_universal_sims_training_pp1+'_50.tsv'
# Feature 4
sim_cse_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/sim_cse_sims_train_pp1.pkl'
sim_cse_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/sim_cse_sims_train_pp1.tsv'
top_n_sim_cse_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/top_n_sim_cse_sims_train_pp1'
top_50_sim_cse_sims_training_pp1_df = top_n_sim_cse_sims_training_pp1+'_50.pkl'
top_50_sim_cse_sims_training_pp1_tsv = top_n_sim_cse_sims_training_pp1+'_50.tsv'
# Feature 5
seq_match_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/seq_match_train_pp1.pkl'
seq_match_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/seq_match_train_pp1.tsv'
# Feature 6
levenshtein_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/levenshtein_train_pp1.pkl'
levenshtein_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/levenshtein_train_pp1.tsv'
# Feature 7
jacc_chars_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/jacc_chars_train_pp1.pkl'
jacc_chars_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/jacc_chars_train_pp1.tsv'
# Feature 8
jacc_tokens_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/jacc_tokens_train_pp1.pkl'
jacc_tokens_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/jacc_tokens_train_pp1.tsv'
# Feature 9
ne_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/ne_sims_train_pp1.pkl'
ne_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/ne_sims_train_pp1.tsv'
# Feature 10
main_syms_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/main_syms_sims_train_pp1.pkl'
main_syms_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/main_syms_sims_train_pp1.tsv'
# Feature 11
words_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/words_sims_train_pp1.pkl'
words_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/words_sims_train_pp1.tsv'
# Feature 12
subjects_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/subjects_sims_train_pp1.pkl'
subjects_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/subjects_sims_train_pp1.tsv'
# Feature 13
token_number_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/token_number_sims_training_pp1.pkl'
token_number_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/token_number_sims_training_pp1.tsv'
# Feature 14
ne_ne_ratio_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/ne_ne_ratio_training_pp1.pkl'
ne_ne_ratio_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/ne_ne_ratio_training_pp1.tsv'
# Feature 15
ne_token_ratio_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/ne_token_ratio_training_pp1.pkl'
ne_token_ratio_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/ne_token_ratio_training_pp1.tsv'
# Feature 16
main_syms_token_ratio_sims_training_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/main_syms_token_ratio_training_pp1.pkl'
main_syms_token_ratio_sims_training_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/main_syms_token_ratio_training_pp1.tsv'
# Feature 17
words_token_ratio_sims_pp1 = 'data/feature_sets/sentence_similarities/training/pp1/words_token_ratio_training_pp1.pkl'
words_token_ratio_sims_pp1_tsv = 'data/feature_sets/sentence_similarities/training/pp1/words_token_ratio_training_pp1.tsv'

# Test

# Feature 1
sbert_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/sbert_sims_test_pp1.pkl'
sbert_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/sbert_sims_test_pp1.tsv'
top_n_sbert_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/top_n_sbert_sims_test_pp1'
top_50_sbert_sims_test_pp1_df = top_n_sbert_sims_test_pp1+'_50.pkl'
top_50_sbert_sims_test_pp1_tsv = top_n_sbert_sims_test_pp1+'_50.tsv'
# Feature 2
infersent_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/infersent_sims_test_pp1.pkl'
infersent_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/infersent_sims_test_pp1.tsv'
top_n_infersent_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/top_n_infersent_sims_test_pp1'
top_50_infersent_sims_test_pp1_df = top_n_infersent_sims_test_pp1+'_50.pkl'
top_50_infersent_sims_test_pp1_tsv = top_n_infersent_sims_test_pp1+'_50.tsv'
# Feature 3
universal_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/universal_sims_test_pp1.pkl'
universal_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/universal_sims_test_pp1.tsv'
top_n_universal_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/top_n_universal_sims_test_pp1'
top_50_universal_sims_test_pp1_df = top_n_universal_sims_test_pp1+'_50.pkl'
top_50_universal_sims_test_pp1_tsv = top_n_universal_sims_test_pp1+'_50.tsv'
# Feature 4
sim_cse_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/sim_cse_sims_test_pp1.pkl'
sim_cse_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/sim_cse_sims_test_pp1.tsv'
top_n_sim_cse_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/top_n_sim_cse_sims_test_pp1'
top_50_sim_cse_sims_test_pp1_df = top_n_sim_cse_sims_test_pp1+'_50.pkl'
top_50_sim_cse_sims_test_pp1_tsv = top_n_sim_cse_sims_test_pp1+'_50.tsv'
# Feature 5
seq_match_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/seq_match_test_pp1.pkl'
seq_match_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/seq_match_test_pp1.tsv'
# Feature 6
levenshtein_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/levenshtein_test_pp1.pkl'
levenshtein_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/levenshtein_test_pp1.tsv'
# Feature 7
jacc_chars_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/jacc_chars_test_pp1.pkl'
jacc_chars_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/jacc_chars_test_pp1.tsv'
# Feature 8
jacc_tokens_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/jacc_tokens_test_pp1.pkl'
jacc_tokens_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/jacc_tokens_test_pp1.tsv'
# Feature 9
ne_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/ne_sims_test_pp1.pkl'
ne_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/ne_sims_test_pp1.tsv'
# Feature 10
main_syms_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/main_syms_sims_test_pp1.pkl'
main_syms_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/main_syms_sims_test_pp1.tsv'
# Feature 11
words_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/words_sims_test_pp1.pkl'
words_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/words_sims_test_pp1.tsv'
# Feature 12
subjects_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/subjects_sims_test_pp1.pkl'
subjects_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/subjects_sims_test_pp1.tsv'
# Feature 13
token_number_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/token_number_sims_test_pp1.pkl'
token_number_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/token_number_sims_test_pp1.tsv'
# Feature 14
ne_ne_ratio_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/ne_ne_ratio_test_pp1.pkl'
ne_ne_ratio_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/ne_ne_ratio_test_pp1.tsv'
# Feature 15
ne_token_ratio_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/ne_token_ratio_test_pp1.pkl'
ne_token_ratio_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/ne_token_ratio_test_pp1.tsv'
# Feature 16
main_syms_token_ratio_sims_test_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/main_syms_token_ratio_test_pp1.pkl'
main_syms_token_ratio_sims_test_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/main_syms_token_ratio_test_pp1.tsv'
# Feature 17
words_token_ratio_sims_pp1 = 'data/feature_sets/sentence_similarities/test/pp1/words_token_ratio_test_pp1.pkl'
words_token_ratio_sims_pp1_tsv = 'data/feature_sets/sentence_similarities/test/pp1/words_token_ratio_test_pp1.tsv'

# Combined sentence embedding similarities

## training

train_sbert_infersent_disjunction = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_sbert_infersent_disjunction.pkl'
train_sbert_infersent_universal_disjunction = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_sbert_infersent_universal_disjunction.pkl'
train_sbert_infersent_universal_sim_cse_disjunction = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_sbert_infersent_universal_sim_cse_disjunction.pkl'
train_sbert_infersent_universal_sim_cse_disjunction_tsv = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_sbert_infersent_universal_sim_cse_disjunction.tsv'

## test

test_sbert_infersent_disjunction = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_sbert_infersent_disjunction.pkl'
test_sbert_infersent_universal_disjunction = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_sbert_infersent_universal_disjunction.pkl'
test_sbert_infersent_universal_sim_cse_disjunction = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction.pkl'
test_sbert_infersent_universal_sim_cse_disjunction_tsv = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_sbert_infersent_universal_sim_cse_disjunction.tsv'

# Combined sentence embeddings similarities + other features

## training

train_first_five_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_five_features.pkl'
train_first_six_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_six_features.pkl'
train_first_seven_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_seven_features.pkl'
train_first_eight_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_eight_features.pkl'
train_first_nine_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_nine_features.pkl'
train_first_ten_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_ten_features.pkl'
train_first_eleven_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_eleven_features.pkl'
train_first_twelve_features = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_twelve_features.pkl'

train_first_twelve_features_tsv = 'data/feature_sets/training/pp1/incomplete_feature_sets/train_first_twelve_features.tsv'

## test

test_first_five_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_five_features.pkl'
test_first_six_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_six_features.pkl'
test_first_seven_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_seven_features.pkl'
test_first_eight_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_eight_features.pkl'
test_first_nine_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_nine_features.pkl'
test_first_ten_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_ten_features.pkl'
test_first_eleven_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_eleven_features.pkl'
test_first_twelve_features = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_twelve_features.pkl'

test_first_twelve_features_tsv = 'data/feature_sets/test/pp1/incomplete_feature_sets/test_first_twelve_features.tsv'




