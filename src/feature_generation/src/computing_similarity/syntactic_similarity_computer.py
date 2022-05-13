
class SyntacticSimComputer:

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
        flatten_1 = SyntacticSimComputer.add_flatten_lists(list_of_entities_1)
        flatten_2 = SyntacticSimComputer.add_flatten_lists(list_of_entities_2)
        if len(flatten_2) > 0 and flatten_2[0] is list:
            flatten_1 = SyntacticSimComputer.add_flatten_lists(flatten_1)
            flatten_2 = SyntacticSimComputer.add_flatten_lists(flatten_2)
        a_set = set(flatten_1)
        b_set = set(flatten_2)
        conj = a_set & b_set
        if 'not available' in conj:
            conj.remove('not available')
        all_elements = a_set | b_set
        if all_elements:
            return len(conj)*2 / len(all_elements)
        else:
            return 0
