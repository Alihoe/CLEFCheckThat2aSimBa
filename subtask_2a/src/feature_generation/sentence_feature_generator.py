import pandas as pd


from src.feature_generation import Features, sbert_encodings_vclaims_pp1, sbert_encodings_training_pp1, \
    sbert_encodings_vclaims_pp1_tsv, sbert_encodings_training_pp1_tsv, sbert_encodings_test_pp1, \
    sbert_encodings_test_pp1_tsv, infersent_encodings_vclaims_pp1, infersent_encodings_vclaims_pp1_tsv, \
    infersent_encodings_training_pp1, infersent_encodings_training_pp1_tsv, infersent_encodings_test_pp1, \
    infersent_encodings_test_pp1_tsv, universal_encodings_vclaims_pp1, universal_encodings_vclaims_pp1_tsv, \
    universal_encodings_training_pp1, universal_encodings_training_pp1_tsv, universal_encodings_test_pp1, \
    universal_encodings_test_pp1_tsv, sim_cse_encodings_vclaims_pp1, sim_cse_encodings_vclaims_pp1_tsv, \
    sim_cse_encodings_training_pp1, sim_cse_encodings_training_pp1_tsv, sim_cse_encodings_test_pp1, \
    sim_cse_encodings_test_pp1_tsv, ne_vclaims_pp1, ne_vclaims_pp1_tsv, ne_training_pp1, ne_training_pp1_tsv, \
    ne_test_pp1, ne_test_pp1_tsv, main_syms_vclaims_pp1, main_syms_vclaims_pp1_tsv, main_syms_training_pp1, \
    main_syms_training_pp1_tsv, main_syms_test_pp1, main_syms_test_pp1_tsv, words_vclaims_pp1, words_vclaims_pp1_tsv, \
    words_training_pp1, words_training_pp1_tsv, words_test_pp1, words_test_pp1_tsv, subjects_vclaims_pp1, \
    subjects_vclaims_pp1_tsv, subjects_training_pp1, subjects_training_pp1_tsv, subjects_test_pp1, \
    subjects_test_pp1_tsv, token_number_vclaims_pp1, token_number_vclaims_pp1_tsv, token_number_training_pp1, \
    token_number_training_pp1_tsv, token_number_test_pp1, token_number_test_pp1_tsv, v_claims_directory, \
    pp_training_data, pp_old_test_data, pp_TEST_data
from src.feature_generation.file_paths.TEST.TEST_file_names import sbert_encodings_training_TEST, \
    sbert_encodings_training_TEST_tsv, sbert_encodings_test_TEST, sbert_encodings_test_TEST_tsv, \
    infersent_encodings_training_TEST, infersent_encodings_training_TEST_tsv, infersent_encodings_test_TEST, \
    infersent_encodings_test_TEST_tsv, universal_encodings_training_TEST, universal_encodings_training_TEST_tsv, \
    universal_encodings_test_TEST, universal_encodings_test_TEST_tsv, sim_cse_encodings_training_TEST, \
    sim_cse_encodings_training_TEST_tsv, sim_cse_encodings_test_TEST, sim_cse_encodings_test_TEST_tsv, ne_training_TEST, \
    ne_training_TEST_tsv, ne_test_TEST, ne_test_TEST_tsv, main_syms_training_TEST, main_syms_training_TEST_tsv, \
    main_syms_test_TEST, main_syms_test_TEST_tsv, words_training_TEST, words_training_TEST_tsv, words_test_TEST, \
    words_test_TEST_tsv, subjects_training_TEST, subjects_training_TEST_tsv, subjects_test_TEST, subjects_test_TEST_tsv, \
    token_number_training_TEST, token_number_training_TEST_tsv, token_number_test_TEST, token_number_test_TEST_tsv
from src.feature_generation.file_paths.pp2.pp2_files import sbert_encodings_training_pp2, \
    sbert_encodings_training_pp2_tsv, \
    sbert_encodings_test_pp2, sbert_encodings_test_pp2_tsv, infersent_encodings_training_pp2, \
    infersent_encodings_training_pp2_tsv, \
    infersent_encodings_test_pp2, infersent_encodings_test_pp2_tsv, universal_encodings_training_pp2, \
    universal_encodings_training_pp2_tsv, \
    universal_encodings_test_pp2, universal_encodings_test_pp2_tsv, sim_cse_encodings_training_pp2, \
    sim_cse_encodings_training_pp2_tsv, \
    sim_cse_encodings_test_pp2, sim_cse_encodings_test_pp2_tsv, ne_training_pp2, \
    ne_training_pp2_tsv, ne_test_pp2, ne_test_pp2_tsv, main_syms_training_pp2, main_syms_training_pp2_tsv, \
    main_syms_test_pp2, main_syms_test_pp2_tsv, \
    words_training_pp2, words_training_pp2_tsv, words_test_pp2, words_test_pp2_tsv, \
    subjects_training_pp2, subjects_training_pp2_tsv, subjects_test_pp2, \
    subjects_test_pp2_tsv, token_number_training_pp2, token_number_training_pp2_tsv, token_number_test_pp2, \
    token_number_test_pp2_tsv
from src.feature_generation.src.creating_datafiles.data_encoder import DataEncoder
from src.feature_generation.src.creating_datafiles.data_lexeme_finder import DataLexemeFinder
from src.feature_generation.src.creating_datafiles.data_named_entity_disambiguator import DataNamedEntityDisambiguator
from src.feature_generation.src.creating_datafiles.data_syntactic_sim_finder import DataSyntacticInfoFinder
from src.feature_generation.src.finding_synonyms.lexeme_finder import LexemeFinder


class SentenceFeatureGenerator:

    @staticmethod
    def create_sentence_features(list_of_features, data_set):
        if Features.sbert.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = sbert_encodings_vclaims_pp1
                    filename_tsv = sbert_encodings_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = sbert_encodings_training_pp1
                    filename_tsv = sbert_encodings_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = sbert_encodings_training_pp2
                        filename_tsv = sbert_encodings_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = sbert_encodings_training_TEST
                        filename_tsv = sbert_encodings_training_TEST_tsv
                elif 'test' in data_set:
                    filename = sbert_encodings_test_pp1
                    filename_tsv = sbert_encodings_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = sbert_encodings_test_pp2
                        filename_tsv = sbert_encodings_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = sbert_encodings_test_TEST
                        filename_tsv = sbert_encodings_test_TEST_tsv
                data_encoder = DataEncoder('sbert_encoder', 'large_model', 'full_text')
                data_encoder.encode(data_set, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong encoding with sbert.')
        if Features.infersent.name in list_of_features: #InferSent - Encodings have to be done for all encodings at once
            try:
                dataset_vclaims = v_claims_directory
                dataset_train_pp1 = pp_training_data
                dataset_test_pp1 = pp_old_test_data
                dataset_test_TEST = pp_TEST_data
                filename_vclaims = infersent_encodings_vclaims_pp1
                filename_vclaims_tsv = infersent_encodings_vclaims_pp1_tsv
                filename_train_pp1 = infersent_encodings_training_pp1
                filename_train_pp1_tsv = infersent_encodings_training_pp1_tsv
                filename_test_pp1 = infersent_encodings_test_pp1
                filename_test_pp1_tsv = infersent_encodings_test_pp1_tsv
                filename_test_TEST = infersent_encodings_test_TEST
                filename_test_TEST_tsv = infersent_encodings_test_TEST_tsv

                sentences_for_vocab = []
                sentences_for_vocab.extend(DataEncoder.get_sentences_to_encode(dataset_vclaims))
                sentences_for_vocab.extend(DataEncoder.get_sentences_to_encode(dataset_train_pp1))
                sentences_for_vocab.extend(DataEncoder.get_sentences_to_encode(dataset_test_pp1))
                sentences_for_vocab.extend(DataEncoder.get_sentences_to_encode(dataset_test_TEST))
                data_encoder = DataEncoder('infer_sent_encoder', 'fast_text_embeddings', 'full_text', sentences_for_vocab)

                data_encoder.encode(dataset_vclaims, filename_vclaims)
                pd.read_pickle(filename_vclaims).to_csv(filename_vclaims_tsv)
                data_encoder.encode(dataset_train_pp1, filename_train_pp1)
                pd.read_pickle(filename_train_pp1).to_csv(filename_train_pp1_tsv)
                data_encoder.encode(dataset_test_pp1, filename_test_pp1)
                pd.read_pickle(filename_test_pp1).to_csv(filename_test_pp1_tsv)
                data_encoder.encode(dataset_test_TEST, filename_test_TEST)
                pd.read_pickle(filename_test_TEST).to_csv(filename_test_TEST_tsv)
            except RuntimeError:
                print('Something went wrong encoding with infersent.')
        if Features.universal.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = universal_encodings_vclaims_pp1
                    filename_tsv = universal_encodings_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = universal_encodings_training_pp1
                    filename_tsv = universal_encodings_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = universal_encodings_training_pp2
                        filename_tsv = universal_encodings_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = universal_encodings_training_TEST
                        filename_tsv = universal_encodings_training_TEST_tsv
                elif 'test' in data_set:
                    filename = universal_encodings_test_pp1
                    filename_tsv = universal_encodings_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = universal_encodings_test_pp2
                        filename_tsv = universal_encodings_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = universal_encodings_test_TEST
                        filename_tsv = universal_encodings_test_TEST_tsv
                data_encoder = DataEncoder('universal_sentence_encoder', '', 'full_text')
                data_encoder.encode(data_set, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong encoding with universal sentence encoder.')
        if Features.sim_cse.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = sim_cse_encodings_vclaims_pp1
                    filename_tsv = sim_cse_encodings_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = sim_cse_encodings_training_pp1
                    filename_tsv = sim_cse_encodings_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = sim_cse_encodings_training_pp2
                        filename_tsv = sim_cse_encodings_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = sim_cse_encodings_training_TEST
                        filename_tsv = sim_cse_encodings_training_TEST_tsv
                elif 'test' in data_set:
                    filename = sim_cse_encodings_test_pp1
                    filename_tsv = sim_cse_encodings_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = sim_cse_encodings_test_pp2
                        filename_tsv = sim_cse_encodings_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = sim_cse_encodings_test_TEST
                        filename_tsv = sim_cse_encodings_test_TEST_tsv
                data_encoder = DataEncoder('sim_cse_encoder', 'large_model', 'full_text')
                data_encoder.encode(data_set, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong encoding with sim cse.')
        if Features.ne.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = ne_vclaims_pp1
                    filename_tsv = ne_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = ne_training_pp1
                    filename_tsv = ne_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = ne_training_pp2
                        filename_tsv = ne_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = ne_training_TEST
                        filename_tsv = ne_training_TEST_tsv
                elif 'test' in data_set:
                    filename = ne_test_pp1
                    filename_tsv = ne_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = ne_test_pp2
                        filename_tsv = ne_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = ne_test_TEST
                        filename_tsv = ne_test_TEST_tsv
                ne_dis = DataNamedEntityDisambiguator()
                ne_dis.disambiguate_named_entities(data_set, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with entity fishing.')
        if Features.main_syms.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = main_syms_vclaims_pp1
                    filename_tsv = main_syms_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = main_syms_training_pp1
                    filename_tsv = main_syms_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = main_syms_training_pp2
                        filename_tsv = main_syms_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = main_syms_training_TEST
                        filename_tsv = main_syms_training_TEST_tsv
                elif 'test' in data_set:
                    filename = main_syms_test_pp1
                    filename_tsv = main_syms_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = main_syms_test_pp2
                        filename_tsv = main_syms_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = main_syms_test_TEST
                        filename_tsv = main_syms_test_TEST_tsv
                lex_finder = LexemeFinder('main_synonyms')
                DataLexemeFinder.find_all_lexemes(data_set, filename, lex_finder)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with finding synonyms.')
        if Features.words.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = words_vclaims_pp1
                    filename_tsv = words_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = words_training_pp1
                    filename_tsv = words_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = words_training_pp2
                        filename_tsv = words_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = words_training_TEST
                        filename_tsv = words_training_TEST_tsv
                elif 'test' in data_set:
                    filename = words_test_pp1
                    filename_tsv = words_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = words_test_pp2
                        filename_tsv = words_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = words_test_TEST
                        filename_tsv = words_test_TEST_tsv
                lex_finder = LexemeFinder('only_used_words')
                DataLexemeFinder.find_all_lexemes(data_set, filename, lex_finder)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with finding words.')
        if Features.subjects.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = subjects_vclaims_pp1
                    filename_tsv = subjects_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = subjects_training_pp1
                    filename_tsv = subjects_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = subjects_training_pp2
                        filename_tsv = subjects_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = subjects_training_TEST
                        filename_tsv = subjects_training_TEST_tsv
                elif 'test' in data_set:
                    filename = subjects_test_pp1
                    filename_tsv = subjects_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = subjects_test_pp2
                        filename_tsv = subjects_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = subjects_test_TEST
                        filename_tsv = subjects_test_TEST_tsv
                syn_sim_finder = DataSyntacticInfoFinder('subjects')
                syn_sim_finder.find_all_syntact_info(data_set, filename)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with finding subjects.')
        if Features.token_number.name in list_of_features:
            try:
                if 'vclaims' in data_set:
                    filename = token_number_vclaims_pp1
                    filename_tsv = token_number_vclaims_pp1_tsv
                elif 'train' in data_set or 'dev' in data_set:
                    filename = token_number_training_pp1
                    filename_tsv = token_number_training_pp1_tsv
                    if 'pp2' in data_set:
                        filename = token_number_training_pp2
                        filename_tsv = token_number_training_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = token_number_training_TEST
                        filename_tsv = token_number_training_TEST_tsv
                elif 'test' in data_set:
                    filename = token_number_test_pp1
                    filename_tsv = token_number_test_pp1_tsv
                    if 'pp2' in data_set:
                        filename = token_number_test_pp2
                        filename_tsv = token_number_test_pp2_tsv
                    elif 'TEST' in data_set:
                        filename = token_number_test_TEST
                        filename_tsv = token_number_test_TEST_tsv
                lex_finder = LexemeFinder('token_number')
                DataLexemeFinder.find_all_lexemes(data_set, filename, lex_finder)
                pd.read_pickle(filename).to_csv(filename_tsv)
            except RuntimeError:
                print('Something went wrong with finding token_number.')




