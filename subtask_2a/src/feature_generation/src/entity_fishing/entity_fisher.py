from nerd import nerd_client
import regex as re


class EntityFisher:

    def __init__(self):
        self.client = nerd_client.NerdClient()

    @staticmethod
    def get_rid_of_hashtags(tweet):
        hashtags = re.findall("(#[A-Za-z0-9]+)", tweet)
        for hashtag in hashtags:
            new_string = re.sub(r"#", "", hashtag)
            new_string = re.sub(r"(?<=\w)([A-Z])", r" \1", new_string)
            tweet = re.sub(hashtag, new_string, tweet)
        return tweet

    def get_named_entities_of_sentence(self, sentence):
        sentence = EntityFisher.get_rid_of_hashtags(sentence)
        try:
            entities = self.client.disambiguate_text(sentence, language='en')[0]['entities']
        except:
            print('Error occured for: ' + sentence)
            entities = []
        entity_list = []
        for entity in entities:
            name = entity['rawName']
            if 'wikipediaExternalRef' in entity:
                wikipedia_id = entity['wikipediaExternalRef']
            else:
                wikipedia_id = 'not available'
            if 'wikidataId' in entity:
                wikidata_id = entity['wikidataId']
            else:
                wikidata_id = 'not available'
            entity_list.append([name, wikipedia_id, wikidata_id])
        return entity_list


