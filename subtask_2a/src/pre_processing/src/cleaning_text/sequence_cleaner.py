import regex as re
import enum


class CleaningMethod(enum.Enum):
    full_cleaning = 1


class SequenceCleaner:

    def __init__(self, cleaning_method):
        self.cleaning_method = cleaning_method

    def clean(self, sequence):
        if self.cleaning_method == CleaningMethod.full_cleaning:
            sequence = re.sub("([^ ]*\.com[/A-Za-z0-9]*)", "", sequence)  # remove .com url
            sequence = re.sub(r"(?:\http?\://|https?\://|www)\S+", "", sequence)  # remove http url
            sequence = re.sub(r"@", "", sequence)
            hyphen_positions = []
            for i in range(len(sequence)):
                if sequence[i] == '—':
                    hyphen_positions.append(i)
            if hyphen_positions:
                last_hyphen = hyphen_positions[len(hyphen_positions) - 1]
                sequence = sequence[:last_hyphen]
            return sequence
        else:
            sequence = re.sub("([^ ]*\.com[/A-Za-z0-9]*)", "", sequence)  # remove .com url
            sequence = re.sub(r"(?:\http?\://|https?\://|www)\S+", "", sequence)  # remove http url
            sequence = re.sub(r"@", "", sequence)
            hyphen_positions = []
            for i in range(len(sequence)):
                if sequence[i] == '—':
                    hyphen_positions.append(i)
            if hyphen_positions:
                last_hyphen = hyphen_positions[len(hyphen_positions) - 1]
                sequence = sequence[:last_hyphen]
            return sequence



