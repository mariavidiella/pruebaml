import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures, DropDuplicateFeatures
from feature_engine.selection import ProbeFeatureSelection
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class SimpleFilter:

    def __init__(self, variance_threshold=0.0, corr_threshold=0.9):
        self.lowVarianceFilter = VarianceThreshold()
        self.filter_duplicates = DropDuplicateFeatures()
        self.correlated_filter = DropCorrelatedFeatures(threshold=corr_threshold)
        self.advanced_filtered = ProbeFeatureSelection(
            estimator=LinearRegression(),
            scoring="neg_mean_absolute_percentage_error",
            n_probes=3,
            distribution="normal",
        )

    def fit(self, X_data, y_data):
        print(X_data.shape)
        self.lowVarianceFilter.fit(X_data)
        lv = self.lowVarianceFilter.transform(X_data)

        lv_df = pd.DataFrame(
            data=lv,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )
        lv_df = lv_df.loc[:,~lv_df.columns.duplicated()].copy()
        print(lv_df.shape)
        self.filter_duplicates.fit(lv_df)
        no_dup = self.filter_duplicates.transform(lv_df)
        print(no_dup.shape)
        self.correlated_filter.fit(no_dup)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        self.advanced_filtered.fit(not_corr, y_data)

    def transform(self, X_data, y_data):
        X_data = X_data.copy()
        X_data_low = self.lowVarianceFilter.transform(X_data)
        X_data_low_df = pd.DataFrame(
            data=X_data_low,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )   
        X_data_low_df = X_data_low_df.loc[:,~X_data_low_df.columns.duplicated()].copy()
        print(X_data_low_df.shape)
        no_dup = self.filter_duplicates.transform(X_data_low_df)
        print(no_dup.shape)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        X_transformed = self.advanced_filtered.transform(not_corr)
        return X_transformed, y_data
    


class ClusteringFilter:

    def __init__(self, eps=0.5, min_samples=5):
        # Inicializamos el algoritmo DBSCAN con los parámetros dados
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # Creamos un escalador para estandarizar las variables numéricas
        self.scaler = StandardScaler()


    def fit(self, X_data, y_data):

        # Copiamos los datos para no modificarlos directamente
        X = X_data.copy()
        y = y_data.copy()

        # Seleccionar variables numéricas relevantes
        self.columns_to_use = ["Area", "No. of Bedrooms"]
        X_num = X[self.columns_to_use]

        # Escalamos 
        X_scaled = self.scaler.fit_transform(X_num)

        # Aplicamos clustering
        self.dbscan.fit(X_scaled)

        # Guardamos las etiquetas asignadas por DBSCAN (-1 significa outlier)
        self.labels = self.dbscan.labels_

        # Nos quedamos solo con los puntos que NO son outliers
        self.sin_outliers = self.labels != -1

    def transform(self, X_data, y_data):

        X = X_data.copy()
        y = y_data.copy()

        # Escalar el test con el mismo scaler
        x_norm = self.scaler.transform(X[self.columns_to_use])

        # Predecir etiquetas DBSCAN sobre los datos test
        labels = self.dbscan.fit_predict(x_norm)

        # Filtramos outliers
        sin_outliers = labels != -1

        return X[sin_outliers], y[sin_outliers]
