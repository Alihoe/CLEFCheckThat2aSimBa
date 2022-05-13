import nltk


class JacquardDistanceTokenizationComputer:

    @staticmethod
    def comp_similarity(string_1, string_2):
        l1 = nltk.word_tokenize(string_1[0])
        l2 = nltk.word_tokenize(string_2)
        a = set(l1)
        b = set(l2)
        return float(len(a.intersection(b))) / len(a.union(b))

