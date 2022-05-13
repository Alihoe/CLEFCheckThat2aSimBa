from nltk import word_tokenize, pos_tag
import spacy


class PartOfSpeechEvaluator:

    @staticmethod
    def get_part_of_speech_order(sentence):
        tokens = word_tokenize(sentence)
        pos = pos_tag(tokens)
        list_of_pos = list(dict(pos).values())
        return list_of_pos

    @staticmethod
    def get_subject_of_sentence(sentence):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        lower_tokens = []
        for token in sub_toks:
            lower_tokens.append(str(token).lower())
        return lower_tokens

