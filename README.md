# pruebaml
# Práctica 1

Este proyecto contiene una práctica donde he creado mis propias clases de transformación de datos y filtrado, y he entrenado un modelo `GradientBoostingRegressor`.

# Práctica 2

Practica2_opt_hp.ipynb -> Preprocesamiento de datos, optimización de hiperparámetros, configuración del estimador base `GradientBoostingRegressor`, configuración del modelo mapie basado en quantile regressor, para intervalos de confianza del 80, 90 y 99%. Guardamos los modelos para su exportación a un entorno de serving/inferencia.

Practica2_server.py -> API en FastAPI que permite cargar modelos de regresión con intervalos de confianza, procesar entradas desde una interfaz web y devolver predicciones con intervalos del 80%, 90% y 99%
