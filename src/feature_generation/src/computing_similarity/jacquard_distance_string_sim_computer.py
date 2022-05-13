class JacquardDistanceComputer:

    @staticmethod
    def comp_similarity(string_1, string_2):
        a = set(string_1[0])
        b = set(string_2)
        return float(len(a.intersection(b))) / len(a.union(b))

