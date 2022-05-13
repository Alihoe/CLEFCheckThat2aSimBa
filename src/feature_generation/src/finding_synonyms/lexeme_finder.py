import enum

import nltk
import regex as re
from nltk import word_tokenize
from nltk.corpus import wordnet


class LexemeFinderType(enum.Enum):
    all_synonyms = 1
    main_synonyms = 2
    only_used_words = 3
    token_number = 4


class LexemeFinder:

    def __init__(self, lexeme_finder_type):
        if lexeme_finder_type == LexemeFinderType.only_used_words.name:
            self.lexeme_finder_type = LexemeFinderType.only_used_words.name
        elif lexeme_finder_type == LexemeFinderType.all_synonyms.name:
            self.lexeme_finder_type = LexemeFinderType.all_synonyms.name
        elif lexeme_finder_type == LexemeFinderType.main_synonyms.name:
            self.lexeme_finder_type = LexemeFinderType.main_synonyms.name
        elif lexeme_finder_type == LexemeFinderType.token_number.name:
            self.lexeme_finder_type = LexemeFinderType.token_number.name

    @staticmethod
    def get_all_synonyms(list_of_words):
        new_list_of_words = []
        for word in list_of_words:
            synsets = wordnet.synsets(word)
            if synsets:
                for synset in synsets:
                    for element in synset.lemmas():
                        synonym = element.name()
                        synonym = re.sub(r"_", " ", synonym)
                        new_list_of_words.append(synonym)
            else:
                new_list_of_words.append(word[0])
        return new_list_of_words

    @staticmethod
    def get_main_synonyms(list_of_words):
        new_list_of_words = []
        for word in list_of_words:
            synsets = wordnet.synsets(word)
            if synsets:
                for synset in synsets:
                    synset_name = synset.name()
                    try:
                        index = synset_name.index('.')
                    except ValueError:
                        index = len(synset_name)
                    synset_name = synset_name[:index]
                    if synset_name not in new_list_of_words:
                        new_list_of_words.append(synset_name)
            else:
                new_list_of_words.append(word[0])
        return new_list_of_words

    def get_lexemes(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if self.lexeme_finder_type == LexemeFinderType.token_number.name:
            return len(tokens)
        else:
            stopwords = nltk.corpus.stopwords.words("english")
            tokens = [word for word in tokens if word not in stopwords]
            if self.lexeme_finder_type == LexemeFinderType.only_used_words.name:
                return tokens
            elif self.lexeme_finder_type == LexemeFinderType.all_synonyms.name:
                return LexemeFinder.get_all_synonyms(tokens)
            elif self.lexeme_finder_type == LexemeFinderType.main_synonyms.name:
                return LexemeFinder.get_main_synonyms(tokens)


