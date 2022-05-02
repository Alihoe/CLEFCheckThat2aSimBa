from Levenshtein import distance as lev


class LevenshteinDistanceComputer:
    @staticmethod
    def comp_similarity(string_1, string_2):
        return -lev(string_1[0], string_2)

