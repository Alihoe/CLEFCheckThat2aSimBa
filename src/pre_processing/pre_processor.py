import enum
import pandas as pd

from src.pre_processing.src.cleaning_text.sequence_cleaner import SequenceCleaner


class PreprocessingMethod(enum.Enum):
    cleaning_tweets = 1


class PreProcessor:

    def __init__(self, preprocessing_method):
        if preprocessing_method == PreprocessingMethod.cleaning_tweets.name:
            self.preprocessing_method = PreprocessingMethod.cleaning_tweets.name
        else:
            raise ValueError('Choose preprocessing method "cleaning".')

    def pre_process(self, input_file, output_file):
        if self.preprocessing_method == PreprocessingMethod.cleaning_tweets.name:
            preprocessor = SequenceCleaner('full_cleaning')
            i_claim_data = pd.read_csv(input_file, sep='\t', names=['iclaim_id', 'iclaim'], dtype=str)
            for row in i_claim_data.iterrows():
                i_claim_id = row[1][0]
                i_claim = row[1][1]
                cleaned_i_claim = preprocessor.clean(i_claim)
                with open(output_file, 'a', encoding='utf-8') as f:
                    joined_list = "\t".join([i_claim_id, cleaned_i_claim])
                    print(joined_list, file=f)
            return output_file
        else:
            raise RuntimeError('Data Preprocessor not properly initialized.')
