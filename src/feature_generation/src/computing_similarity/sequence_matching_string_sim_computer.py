from difflib import SequenceMatcher


class SequenceMatchingComputer:

    @staticmethod
    def comp_similarity(string_1, string_2):
        sequence_comp = SequenceMatcher(a=string_1[0], b=string_2)
        return sequence_comp.ratio()

