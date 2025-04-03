"""
Modulo para procesar y transformar datos.
Todas las clases tendrán al menos dos métodos:
   - fit() -> para ajustar los parámetros
   - transform() -> para transformar las features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from skrub import GapEncoder
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer, RobustScaler
from skrub import SimilarityEncoder


class ExtendedTransformation:

    def __init__(self, ge_components=50):
        self.imputer = SimpleImputer(strategy="median")
        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.gapEncoder = GapEncoder(n_components=ge_components)
        self.y_Transformer = QuantileTransformer()
        self.area_Transformer = QuantileTransformer()
        self.beds_Transformer = QuantileTransformer()
        self.polyfeatures = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.scaler_y = StandardScaler()
        self.scaler_area = StandardScaler()
        self.scalar_beds = StandardScaler()

    def fit(self, X, y):
        X_data = X.copy()
        y_data = y.copy()
        print("X shape: ", X.shape)
        self.bin_vars_columns = X.columns[4:]
        print("bin_vars_columns shape: ", self.bin_vars_columns.shape)

        # fit impute n beds
        self.beds_feaures = "No. of Bedrooms"
        self.imputer.fit(X_data[[self.beds_feaures]])
        X_data = X_data.replace({9: np.nan})
        X_data[self.bin_vars_columns] = X_data[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        # fit low_cardinality features wiht ohot encoding
        self.low_card_columns = ["city"] + self.bin_vars_columns.to_list()
        print("low_card_columns shape: ", len(self.low_card_columns))  
        self.ohEnconder.fit(X_data[self.low_card_columns])
        self.loc_feature = "Location"

        # fit high_cardinality features.
        self.gapEncoder.fit(X_data[self.loc_feature])

        self.area_feature = "Area"

        # fit Quantile transformation of numerical vars.
        self.y_Transformer.fit(y_data)
        self.area_Transformer.fit(X_data[[self.area_feature]])
        self.beds_Transformer.fit(X_data[[self.beds_feaures]])

        self.scaler_y.fit(self.y_Transformer.transform(y_data))
        self.scaler_area.fit(
            self.area_Transformer.transform(X_data[[self.area_feature]])
        )
        self.scalar_beds.fit(
            self.beds_Transformer.transform(X_data[[self.beds_feaures]])
        )

        # scale to standard

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()

        # impute missing data
        X = X.replace({9: np.nan})
        X[self.bin_vars_columns] = X[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )
        X[self.beds_feaures] = self.imputer.transform(X[[self.beds_feaures]])
        print("X shape: ", X.shape)
        # transform categorical features.
        cat_low_card_tfed = self.ohEnconder.transform(X[self.low_card_columns])
        X_low_card = pd.DataFrame(
            data=cat_low_card_tfed,
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        print("X_low_card   shape: ", X_low_card.shape)

        X_high_card = self.gapEncoder.transform(X[self.loc_feature])
        print("X_high_card shape: ", X_high_card.shape)

        # transform numerical vars.
        y_transformed = self.y_Transformer.transform(y)
        area_normal = self.area_Transformer.transform(X[[self.area_feature]])
        beds_normal = self.beds_Transformer.transform(X[[self.beds_feaures]])

        y_scaled = self.scaler_y.transform(y_transformed)
        area_scaled = self.scaler_area.transform(area_normal)
        beds_scaled = self.scalar_beds.transform(beds_normal)

        X_num = pd.DataFrame(
            data={
                self.area_feature: area_scaled.flatten(),
                self.beds_feaures: beds_scaled.flatten(),
            },
            index=X.index,
        )
        features_to_cross = pd.concat([X_low_card,X_num], axis=1)
        self.polyfeatures.fit(features_to_cross)
        crossed_features = self.polyfeatures.transform(features_to_cross)

        X_crossed_features = pd.DataFrame(
            data=crossed_features,
            columns=self.polyfeatures.get_feature_names_out(),
            index=X.index,
        )
        print("X_crossed_features shape: ", X_crossed_features.shape)
        X_EXPANDED = pd.concat([X_num, X_low_card, X_high_card, X_crossed_features], axis=1)
        print("X_EXPANDED shape: ", X_EXPANDED.shape)
        return X_EXPANDED, y_scaled

    def inverse_transform(self, y_data):
        return self.y_Transformer.inverse_transform(
            self.scaler_y.inverse_transform(y_data)
        )


class SimpleTransformation:

    def fit(self, X_data, y_data):
        self.remove_column = "Location"
        self.impute_columns = list(
            set(X_data.columns.to_list()) - set([self.remove_column, "city"])
        )
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X_data[self.impute_columns])

        self.ohEnconder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohEnconder.fit(X_data[["city"]])

    def transform(self, X_data, y_data):
        X = X_data.copy()
        y = y_data.copy()
        X = X.drop(columns=[self.remove_column])
        X[self.impute_columns] = self.imputer.transform(X[self.impute_columns])
        X_cat = pd.DataFrame(
            data=self.ohEnconder.transform(X_data[["city"]]),
            columns=self.ohEnconder.get_feature_names_out(),
            index=X.index,
        )
        X_final = pd.concat([X.drop(columns=["city"]), X_cat], axis=1)
        return X_final, y
    


class MyTransformation:

    def __init__(self, n_neighbors=5):

        # Imputador para "No. of Bedrooms" usando KNN
        self.imputer_beds = KNNImputer(n_neighbors=n_neighbors)
        # Imputador para "Area" usando la mediana
        self.imputer_area = SimpleImputer(strategy="median")

        # Transformadores para normalizar la distribución de las variables numéricas
        self.beds_transformer = PowerTransformer(method="yeo-johnson")
        self.area_transformer = PowerTransformer(method="yeo-johnson")

        # Escaladores que reducen el efecto de los valores extremos
        self.beds_scaler = RobustScaler()
        self.area_scaler = RobustScaler()

        # Transformador y escalador para la variable objetivo "Price"
        self.y_transformer = PowerTransformer(method="yeo-johnson")
        self.scaler_y = RobustScaler()

        # Codificador para variables categóricas con muchas clases
        self.encoder = SimilarityEncoder()

        # Indicamos si queremos incluir la frecuencia de cada categoría como nueva feature
        self.include_freq = True

    def fit(self, X_data, y_data):

        # Copiamos los datos para no modificarlos directamente
        X = X_data.copy()
        y = y_data.copy()

        # Guardamos los nombres de las columnas clave
        self.bedrooms_column = "No. of Bedrooms"
        self.area_column = "Area"
        self.cat_columns = ["city", "Location"]
        self.bin_vars_columns = X.columns[4:]

        # Reemplazar 9 por NaN ya que indican los valores faltantes
        X = X.replace({9: np.nan})

        # Reemplazar binarios 0/1/NaN por texto
        X[self.bin_vars_columns] = X[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )

        # Ajustamos los imputadores
        self.imputer_area.fit(X[[self.area_column]])
        self.imputer_beds.fit(X[[self.bedrooms_column]])

        # Ajustamos los transformadores para hacer las variables más normales
        self.area_transformer.fit(X[[self.area_column]])
        self.beds_transformer.fit(X[[self.bedrooms_column]])

        # Ajustamos los escaladores para trabajar sobre variables normalizadas
        self.area_scaler.fit(self.area_transformer.transform(X[[self.area_column]]))
        self.beds_scaler.fit(self.beds_transformer.transform(X[[self.bedrooms_column]]))

        # Codificadores categóricos
        self.encoders = {}
        for col in self.cat_columns:
            X[col] = X[col].fillna("NO DISPONIBLE")
            enc = SimilarityEncoder()
            enc.fit(X[[col]])
            self.encoders[col] = enc

        # Ajustamos la transformación y escalado de la variable objetivo
        self.y_transformer.fit(y)
        self.scaler_y.fit(self.y_transformer.transform(y))

    def transform(self, X_data, y_data):

        # Copiamos los datos para no modificarlos directamente
        X = X_data.copy()
        y = y_data.copy()
        
        # Reemplazamos 9 por NaN en caso de que queden. Igual que en fit, convertimos variables binarias a texto
        X = X.replace({9: np.nan})
        X[self.bin_vars_columns] = X[self.bin_vars_columns].replace(
            {0: "NO", 1: "SI", np.nan: "NO_DISPONIBLE"}
        )

        # Imputamos valores faltantes
        X[self.area_column] = self.imputer_area.transform(X[[self.area_column]])
        X[self.bedrooms_column] = self.imputer_beds.transform(X[[self.bedrooms_column]])

        # Transformación y escalado
        area_pt = self.area_transformer.transform(X[[self.area_column]])
        beds_pt = self.beds_transformer.transform(X[[self.bedrooms_column]])

        area_scaled = self.area_scaler.transform(area_pt)
        beds_scaled = self.beds_scaler.transform(beds_pt)

        # Creamos DataFrame con variables numéricas finales
        X_num = pd.DataFrame({
            self.area_column: area_scaled.flatten(),
            self.bedrooms_column: beds_scaled.flatten()
        }, index=X.index)

        # Codificación categórica
        encoded_features = []
        for col in self.cat_columns:
            X[col] = X[col].fillna("MISSING")
            encoded = self.encoders[col].transform(X[[col]])
            encoded_df = pd.DataFrame(encoded, index=X.index)
            encoded_df.columns = [f"{col}_sim_{i}" for i in range(encoded.shape[1])]
            encoded_features.append(encoded_df)

            # Añadimos la frecuencia como nueva columna si está activado
            if self.include_freq:
                freq_map = X[col].value_counts().to_dict()
                X[f"{col}_freq"] = X[col].map(freq_map)

        # Eliminamos las columnas categóricas originales
        X = X.drop(columns=self.cat_columns)

         # Unimos todo: numéricas escaladas + codificadas + frecuencias
        X_final = pd.concat(
            [X_num] + encoded_features + [X[[f"{col}_freq" for col in self.cat_columns]]],
            axis=1
        )

        # Escalamos la variable objetivo
        y_scaled = self.scaler_y.transform(self.y_transformer.transform(y))

        return X_final, y_scaled

    def inverse_transform(self, y_data):
        # Deshacemos el escalado y la transformación de la variable objetivo
        return self.y_transformer.inverse_transform(
            self.scaler_y.inverse_transform(y_data)
        )