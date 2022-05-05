import enum
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn import svm, tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BalancingMethod(enum.Enum):
    random_undersampling = 1
    SMOTE = 2
    near_miss = 3


class DatScaler(enum.Enum):
    standard_scaler = 1
    min_max_scaler = 2
    replace_negatives = 3


class MLModel(enum.Enum):
    log_reg = 1
    svm = 2
    knn = 3
    mn_naive_bayes = 4
    dec_tree = 5
    linear_reg = 6


class RankClassifier:

    def __init__(self, scaler_name='standard_scaler', model_name='log_reg', balancing_name=0, sampling_strat=0.1):
        self.sampling_strat = sampling_strat
        if scaler_name == DatScaler.standard_scaler.name:
            self.scaler = StandardScaler()
        elif scaler_name == DatScaler.min_max_scaler.name:
            self.scaler = MinMaxScaler()
        elif scaler_name == DatScaler.replace_negatives.name:
            self.scaler = DatScaler.replace_negatives.name
        else:
            raise ValueError('Choose "standard_scaler".')
        if balancing_name == BalancingMethod.random_undersampling.name:
            self.balancer = RandomUnderSampler(sampling_strategy=self.sampling_strat, random_state=86)
        elif balancing_name == BalancingMethod.SMOTE.name:
            self.balancer = SMOTE(sampling_strategy=self.sampling_strat)
        elif balancing_name == BalancingMethod.near_miss.name:
            self.balancer = NearMiss()
        elif not balancing_name:
            self.balancer = False
        if model_name == MLModel.log_reg.name:
            self.model = LogisticRegression()
        elif model_name == MLModel.svm.name:
            self.model = svm.SVC()
        elif model_name == MLModel.knn.name:
            self.model = KNeighborsClassifier(n_neighbors=3)
        elif model_name == MLModel.mn_naive_bayes.name:
            self.model = MultinomialNB()
        elif model_name == MLModel.dec_tree.name:
            self.model = tree.DecisionTreeClassifier()
        elif model_name == MLModel.linear_reg.name:
            self.model = LinearRegression()
        else:
            raise ValueError('Choose "log_reg", "svm", "knn", "mn_naive_bayes" or "dec_tree".')

    def training_of_classifier(self, feature_set_path):
        if isinstance(feature_set_path, str):
            data_set = pd.read_pickle(feature_set_path)
        else:
            data_set = feature_set_path
        print('Positives', round(data_set['score'].value_counts()[1] / len(data_set) * 100, 2), '% of the dataset')
        col = data_set.columns
        x = data_set.iloc[:, 2:len(col)-1]
        x_col = x.columns
        y = data_set.iloc[:, len(col)-1]
        if self.scaler == DatScaler.replace_negatives.name:
            features = x.columns
            new_x = pd.DataFrame()
            for feature in features:
                print(feature)
                this_feature = x[feature]
                print(this_feature[this_feature < 0])
                negative_values = len(this_feature[this_feature < 0])
                print(negative_values)
                if negative_values <= len(this_feature):
                    print('more positive values')
                    this_feature[this_feature < 0] = 0
                else:
                    print('more negative values')
                    this_feature[this_feature > 0] = 0
                new_x[feature] = this_feature
        else:
            x = self.scaler.fit_transform(x)
        if self.balancer:
            x, y = self.balancer.fit_resample(x, y)
        selector = SelectFromModel(estimator=LogisticRegression()).fit(x, y)
        print(selector.get_feature_names_out(x_col))
        x = selector.transform(x)
        self.model.fit(x, y)
        return self.model, selector

    def predict_rankings(self, model, feature_set_without_rankings, selector):
        if isinstance(feature_set_without_rankings, str):
            data_set = pd.read_pickle(feature_set_without_rankings)
        else:
            data_set = feature_set_without_rankings
        col = data_set.columns
        x = data_set.iloc[:, 2:len(col)]
        if self.scaler == DatScaler.replace_negatives.name:
            features = x.columns
            new_x = pd.DataFrame()
            for feature in features:
                print(feature)
                this_feature = x[feature]
                print(this_feature[this_feature < 0])
                negative_values = len(this_feature[this_feature < 0])
                print(negative_values)
                if negative_values <= len(this_feature):
                    print('more positive values')
                    this_feature[this_feature < 0] = 0
                else:
                    print('more negative values')
                    this_feature[this_feature > 0] = 0
                new_x[feature] = this_feature
        else:
            x = self.scaler.fit_transform(x)
        x = selector.transform(x)
        prediction = model.predict(x)
        data_set['rank'] = prediction
        return data_set

    def training_of_prediction_classifier(self, feature_set_path):
        if isinstance(feature_set_path, str):
            data_set = pd.read_pickle(feature_set_path)
        else:
            data_set = feature_set_path
        print('Positives', round(data_set['score'].value_counts()[1] / len(data_set) * 100, 2), '% of the dataset')
        col = data_set.columns
        x = data_set.iloc[:, 2:len(col)-1]
        y = data_set.iloc[:, len(col)-1]
        x = self.scaler.fit_transform(x)
        if self.balancer:
            x, y = self.balancer.fit_resample(x, y)
        calibrated_clf = CalibratedClassifierCV(base_estimator=self.model)
        calibrated_clf.fit(x, y)
        return calibrated_clf

    def predict_probabilities(self, model, feature_set_without_rankings):
        if isinstance(feature_set_without_rankings, str):
            data_set = pd.read_pickle(feature_set_without_rankings)
        else:
            data_set = feature_set_without_rankings
        col = data_set.columns
        x = data_set.iloc[:, 2:len(col)]
        x = self.scaler.fit_transform(x)
        prediction = model.predict_proba(x)
        prediction = [i[1] for i in list(prediction)]
        data_set['rank'] = prediction
        return data_set

