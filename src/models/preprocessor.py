import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.pipeline import Pipeline, FunctionTransformer
from typing import Callable
from sklearn.compose import ColumnTransformer


class Preprocessor:

    def _get_feature_selector(self) -> FunctionTransformer:
        if self.return_df:
            selector = lambda df: df[self.features['all']]
        else:
            selector = lambda df: df[self.features['all']].values
        return FunctionTransformer(selector)

    @staticmethod
    def validate_features(key_outter: str, key_inner: str) -> Callable | None:
        def decorator(func):
            def validator(self):
                if key_outter in self.features.keys():
                    if key_inner in self.features[key_outter].keys():
                        if len(self.features[key_outter][key_inner]) > 0:
                            columns = [
                                column
                                for column in self.features[key_outter][key_inner]
                                if column in self.features['all']
                            ]
                            if len(columns) == 0:
                                return None
                            return func(self, columns=columns)
                return None
            return validator
        return decorator

    @validate_features('encoders', 'catboost')
    def _get_catboost_enc(self, **kwargs) -> CatBoostEncoder | None:
        columns = kwargs['columns']
        if not self.return_df:
            columns = [
                self.features['all'].index(column)
                for column in columns
            ]
        return CatBoostEncoder(
                    verbose=0,
                    cols=columns,
                    handle_unknown='values',
                    handle_missing='return_nan',
                    a=1,
                    return_df=self.return_df,
                    random_state=0
                )

    @validate_features('encoders', 'ordinal')
    def _get_ordinal_enc(self, **kwargs) -> OrdinalEncoder | None:
        columns = kwargs['columns']
        if not self.return_df:
            columns = [
                self.features['all'].index(column)
                for column in columns
            ]
        return OrdinalEncoder(
                verbose=0,
                cols=columns,
                handle_unknown='value',
                handle_missing='return_nan',
                return_df=self.return_df
            )

    @validate_features('imputers', 'simple')
    def _get_simple_imputer(self, **kwargs) -> Pipeline:
        columns = kwargs['columns']
        if not self.return_df:
            columns = [
                self.features['all'].index(column)
                for column in columns
            ]
        imputer = SimpleImputer(strategy=self.imputer_strategy)
        imputer = ColumnTransformer(
                    transformers=[
                        ('imputer', imputer, columns)
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )
        pipline = [('imputer', imputer)]

        if self.return_df:
            def make_df_func(data):
                if self.feature_names_out is not None:
                    return pd.DataFrame(
                            data, columns=self.feature_names_out
                        )[self.features['all']]
                return pd.DataFrame(
                            data, columns=imputer.get_feature_names_out()
                        )[self.features['all']]
            make_df = FunctionTransformer(make_df_func)
            pipline.append(('make_df', make_df))
            return Pipeline(pipline)

        def sort_array_func(data):
            sort = lambda s: int(s[1::])
            if self.feature_names_out is not None:
                return lambda data: data[:, np.vectorize(sort)(self.feature_names_out)]
            return lambda data: data[:, np.vectorize(sort)(imputer.get_feature_names_out())]
        sort_array = FunctionTransformer(sort_array_func)
        pipline.append(('sort_array', sort_array))
        return Pipeline(pipline)

    def __init__(
            self,
            features: dict = {},
            imputer_strategy: str | None = None,
            simple_impute_all: bool = False,
            return_df: bool = False
        ) -> None:

        self.features = features
        self.imputer_strategy = imputer_strategy
        self.return_df = return_df
        self.feature_names_out = None
        if not features:
            return None

        feature_selector = self._get_feature_selector()
        self.pipeline = [('feature_selection', feature_selector)]

        catboost_encoder = self._get_catboost_enc()
        if catboost_encoder:
            self.pipeline.append(('catboost', catboost_encoder))

        ordinal_encoder = self._get_ordinal_enc()
        if ordinal_encoder:
            self.pipeline.append(('ordinal', ordinal_encoder))

        if self.imputer_strategy:
            if simple_impute_all:
                if 'imputers' not in self.features.keys():
                    self.features['imputers'] = dict()
                self.features['imputers']['simple'] = self.features['all']
            simple_imputer = self._get_simple_imputer()
            if simple_imputer:
                self.pipeline.append((f'{self.imputer_strategy}_imputer', simple_imputer))

        self.pipeline = Pipeline(self.pipeline)

    def fit(self, X, y) -> Pipeline:
        fit_pipe = self.pipeline.fit(X, y)
        if 'imputers' in self.features.keys():
            if 'simple' in self.features['imputers'].keys():
                self.feature_names_out = self.pipeline\
                    .named_steps[f'{self.imputer_strategy}_imputer']\
                        .named_steps['imputer'].get_feature_names_out()
        return fit_pipe

    def transform(self, X) -> np.ndarray | pd.DataFrame:
        return self.pipeline.transform(X)

    def fit_transform(self, X, y) -> np.ndarray | pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)