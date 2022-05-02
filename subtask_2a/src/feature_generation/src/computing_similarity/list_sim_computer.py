
class ListEntitySimComputer:

    @staticmethod
    def add_flatten_lists(the_lists):
        result = []
        for _list in the_lists:
            result += _list
        return result

    @staticmethod
    def comp_similarity(list_of_entities_1, list_of_entities_2):
        list_of_entities_1 = list(list_of_entities_1)
        list_of_entities_2 = list(list_of_entities_2)
        flatten_1 = ListEntitySimComputer.add_flatten_lists(list_of_entities_1)
        flatten_2 = ListEntitySimComputer.add_flatten_lists(list_of_entities_2)
        if flatten_1 and isinstance(flatten_1[0], list):
            flatten_1 = ListEntitySimComputer.add_flatten_lists(flatten_1)
        if flatten_2 and isinstance(flatten_2[0], list):
            flatten_2 = ListEntitySimComputer.add_flatten_lists(flatten_2)
        a_set = set(flatten_1)
        b_set = set(flatten_2)
        conj = a_set & b_set
        if 'not available' in conj:
            conj.remove('not available')
        sim = len(conj)
        return sim

    @staticmethod
    def comp_ratio(list_of_entities_1, list_of_entities_2):
        list_of_entities_1 = list(list_of_entities_1)
        list_of_entities_2 = list(list_of_entities_2)
        flatten_1 = ListEntitySimComputer.add_flatten_lists(list_of_entities_1)
        flatten_2 = ListEntitySimComputer.add_flatten_lists(list_of_entities_2)
        if flatten_1 and isinstance(flatten_1[0], list):
            flatten_1 = ListEntitySimComputer.add_flatten_lists(flatten_1)
        if flatten_2 and isinstance(flatten_2[0], list):
            flatten_2 = ListEntitySimComputer.add_flatten_lists(flatten_2)
        a_set = set(flatten_1)
        if 'not available' in a_set:
            a_set.remove('not available')
        b_set = set(flatten_2)
        if 'not available' in b_set:
            b_set.remove('not available')
        union = a_set | b_set
        conj = a_set & b_set
        if len(union) == 0:
            sim = 0
        else:
            sim = (100/len(union))*len(conj)
        return sim


