import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import mutual_info_regression as MIR
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt


class FeatureSelector:

    @staticmethod
    def feature_correlation_for_pairwise_data_with_feature_n(feature_set_path, n):
        data_set = pd.read_pickle(feature_set_path)
        print(data_set)
        col = data_set.columns
        print(col)
        print(col[1:len(col)])
        target_name = data_set.iloc[:, n].name
        print(target_name)
        # cor = data_set.corr(method='pearson')
        # print('pearson')
        # for col_name in col[1:len(col)]:
        #     print(col_name)
        #     print(cor.loc[col_name, target_name])
        cor = data_set.corr(method='spearman')
        print('spearman')
        for col_name in col[1:len(col)]:
            print(col_name)
            print(cor.loc[col_name, target_name])
        # cor = data_set.corr(method='kendall')
        # print('kendall')
        # for col_name in col[1:len(col)]:
        #     print(col_name)
        #     print(cor.loc[col_name, target_name])

    @staticmethod
    def feature_correlation(feature_set_path, output):
        data_set = pd.read_pickle(feature_set_path)
        print(data_set.columns)
        data_set = data_set[['sbert', 'infersent', 'universal', 'sim_cse', 'levenshtein', 'jacc_chars',
                               'jacc_tokens', 'seq_match', 'words', 'words_ratio', 'words_token_ratio', 'main_syms',
                            'main_syms_ratio', 'main_syms_token_ratio',  'ne', 'ne_ne_ratio', 'ne_token_ratio']]
        data_set['seq_match'] = data_set['seq_match'].astype('float')
        data_set['levenshtein'] = data_set['levenshtein'].astype('float')
        data_set['jacc_chars'] = data_set['jacc_chars'].astype('float')
        data_set['jacc_tokens'] = data_set['jacc_tokens'].astype('float')
        data_set['ne'] = data_set['ne'].astype('float')
        data_set['main_syms'] = data_set['main_syms'].astype('float')
        data_set['words'] = data_set['words'].astype('float')
        #data_set['subjects'] = data_set['subjects'].astype('float')
        data_set['ne_ne_ratio'] = data_set['ne_ne_ratio'].astype('float')
        data_set['ne_token_ratio'] = data_set['ne_token_ratio'].astype('float')
        data_set['main_syms_ratio'] = data_set['main_syms_ratio'].astype('float')
        data_set['main_syms_token_ratio'] = data_set['main_syms_token_ratio'].astype('float')
        data_set['words_ratio'] = data_set['words_ratio'].astype('float')
        data_set['words_token_ratio'] = data_set['words_token_ratio'].astype('float')

        cor = data_set.corr(method='spearman').round(decimals=2)
        print(cor)
        cor.to_csv(output, sep='&')


    # Mutual Information
    @staticmethod
    def mutual_information_feature_selection(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        data_set.columns = ['i_claim_id', 'ver_claim_id', 'sbert', 'infersent', 'universal', 'sim_cse',
                            'sequence_matcher', 'levenshtein', 'jacc_char', 'jacc_tok', 'ne', 'main_syns', 'words', 'subjects', 'score']
        data_set['levenshtein'] = data_set['levenshtein'].astype('float')
        data_set['ne'] = data_set['ne'].astype('float')
        data_set['jacc_char'] = data_set['jacc_char'].astype('float')
        data_set['jacc_tok'] = data_set['jacc_tok'].astype('float')
        data_set['main_syns'] = data_set['main_syns'].astype('float')
        data_set['words'] = data_set['words'].astype('float')
        data_set['subjects'] = data_set['subjects'].astype('float')
        col = data_set.columns
        x = data_set.iloc[:, 2:len(col)-1]
        y = data_set.iloc[:, len(col)-1]
        mi_score = MIC(x, y)
        target_mi_dic = {}
        i = 0
        for col_name in col[2:len(col)-1]:
            value = round(mi_score[i], 5)
            target_mi_dic.update({col_name: value})
            print(col_name)
            print(value)
            i = i+1
        return sorted(target_mi_dic, key=target_mi_dic.get, reverse=True)

    # Anova
    @staticmethod
    def anova_feature_selection(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        x = data_set.iloc[:, 1:len(col) - 1]
        print(x)
        y = data_set.iloc[:, len(col) - 1]
        for i in range(len(col)-2):
            fvalue_Best = SelectKBest(f_classif, k=i+1)
            X_kbest = fvalue_Best.fit_transform(x, y)
            # relevant_feature = x.columns[(x.values == X_kbest[:, None]).all(0)]
            print(X_kbest)
           # print(relevant_feature)


    @staticmethod
    def aggregate_rankings(list_of_list_of_rankings):
        first_list = list(reversed(list_of_list_of_rankings[0]))
        lengths = len(first_list)
        rankings_dic = {first_list[i]: i for i in range(0, len(first_list), 1)}
        for ranking_list in list_of_list_of_rankings[1:]:
            if len(ranking_list) != lengths:
                raise ValueError('One list has a different length than the other(s).')
            reversed_ranking_list = list(reversed(ranking_list))
            for i in range(len(reversed_ranking_list)):
                feature = reversed_ranking_list[i]
                old_value = rankings_dic[feature]
                rankings_dic.update({feature: i+old_value})
        print(rankings_dic)
        return sorted(rankings_dic, key=rankings_dic.get, reverse=True)


    # pearson correlation coefficient
    @staticmethod
    def pearson_correlation_feature_selection(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        target_name = data_set.iloc[:, len(col) - 1].name
        # Correlation with output variable
        cor = data_set.corr(method= 'pearson')
        target_cor_dic = {}
        for col_name in col[1:len(col)-1]:
            cor_value = cor.loc[col_name, target_name]
            target_cor_dic.update({col_name: cor_value})
            print(col_name)
            print(cor_value)
        return sorted(target_cor_dic, key=target_cor_dic.get, reverse=True)

    # Spearman's rank correlation coefficient
    @staticmethod
    def spearmans_rank_correlation_coefficient(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        target_name = data_set.iloc[:, len(col) - 1].name
        # Correlation with output variable
        cor = data_set.corr(method='spearman')
        target_cor_dic = {}
        for col_name in col[1:len(col)-1]:
            cor_value = cor.loc[col_name, target_name]
            target_cor_dic.update({col_name: cor_value})
            print(col_name)
            print(cor_value)
        return sorted(target_cor_dic, key=target_cor_dic.get, reverse=True)


    # Kendall's rank correlation coefficient
    @staticmethod
    def kendalls_rank_correlation_coefficient(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        target_name = data_set.iloc[:, len(col) - 1].name
        # Correlation with output variable
        cor = data_set.corr(method='kendall')
        target_cor_dic = {}
        for col_name in col[1:len(col)-1]:
            cor_value = cor.loc[col_name, target_name]
            target_cor_dic.update({col_name: cor_value})
            print(col_name)
            print(cor_value)
        return sorted(target_cor_dic, key=target_cor_dic.get, reverse=True)

    # Forward selection
    @staticmethod
    def forward_selection_log_reg(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        number_of_features = len(col)-2
        x = data_set.iloc[:, 1:len(col) - 1]
        x = StandardScaler().fit_transform(x)
        y = data_set.iloc[:, len(col) - 1]
        for number in range(number_of_features):
            sfs = SFS(LogisticRegression(),
                      k_features=number+1,
                      forward=True,
                      floating=False,
                      scoring='accuracy',
                      cv=0)
            sfs.fit(x, y)
            print(number)
            print(sfs.k_feature_names_)

    # Backward elimination
    @staticmethod
    def backward_elimination_log_reg(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        print(col)
        number_of_features = len(col) - 2
        x = data_set.iloc[:, 1:len(col) - 1]
        y = data_set.iloc[:, len(col) - 1]
        x = StandardScaler().fit_transform(x)
        for number in range(number_of_features):
            sbs = SFS(LogisticRegression(),
                      k_features=number+1,
                      forward=False,
                      floating=False,
                      cv=0)
            sbs.fit(x, y)
            print(number+1)
            print(sbs.k_feature_names_)

    @staticmethod
    def sequential_elimination_log_reg(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        print(col)
        number_of_features = len(col) - 2
        x = data_set.iloc[:, 1:len(col) - 1]
        x = StandardScaler().fit_transform(x)
        y = data_set.iloc[:, len(col) - 1]
        # Sequential Forward Floating Selection(sffs)
        for number in range(number_of_features):
            sffs = SFS(LogisticRegression(),
                       k_features=(number+1, number_of_features-number),
                       forward=True,
                       floating=True,
                       cv=0)
            sffs.fit(x, y)
            print(number+1)
            print(number_of_features-number)
            print(sffs.k_feature_names_)

    @staticmethod
    def lasso(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        col_without_id = data_set.columns[1:]
        print(col)
        x = data_set.iloc[:, 1:len(col) - 1]
        print(x)
        y = data_set.iloc[:, len(col) - 1]
        print(y)
        x = StandardScaler().fit_transform(x)
        skf = StratifiedKFold()
        lasso = LassoCV(cv=skf, random_state=2).fit(x, y)
        print(lasso.coef_)
        print('Selected Features:', list(col_without_id[np.where(lasso.coef_ > 0.01)[0]]))

    @staticmethod
    def feature_importance(feature_set_path):
        data_set = pd.read_pickle(feature_set_path)
        col = data_set.columns
        print(col)
        x = data_set.iloc[:, 1:len(col) - 1]
        y = data_set.iloc[:, len(col) - 1]
        x = StandardScaler().fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators = 100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="lightsalmon", align="center")
        plt.xticks(range(X_train.shape[1]), data_set.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

        


