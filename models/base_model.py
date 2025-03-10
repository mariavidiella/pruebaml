import pandas as pd # type: ignore
import numpy as np # type: ignore

class BaseDataProcessing():
    """
    Una clase para procesar los datos de mi modelo base
    """

    def __init__(self):
        self.base_data = None

    def size(self):
        if not self.base_data:
            return None
        return self.base_data.shape[0]
    
    def transform(self, my_data):
        data = my_data.copy()
        data.loc[:,'Standard_location'] = ( data['Location'].
                                        str.lower().
                                        str.replace(" ","").
                                        str.replace(r'\d+',"", regex = True).
                                        str.replace('sector',"") )
        data['price_per_ft^2'] = data['Price'] / data['Area']
        return data 
    
class BaseModel():
    """
    Modelo que estima el precio de un piso en función del precio medio 
    por metro cuadrado de su ciudad y localización
    """

    def __init__(self):
         self.df_summary_city = None
         self.data = None
         self.city_average_dict = None
         self.df_summary = None
         self.city_location_dict = None

    def fit(self, data):
        self.data = data
        self.df_summary_city = data.groupby(['city'])[['price_per_ft^2']].mean()
        self.city_average_dict = self.df_summary_city.to_dict()['price_per_ft^2']
        self.df_summary = data.groupby(['city','Standard_location'])[['price_per_ft^2']].mean()
        self.city_location_dict = self.df_summary.to_dict()['price_per_ft^2']

    def predict(self, data):
        copy_data = data.copy()

        def calculare_price(x):
            city_average = self.city_average_dict[x['city']]
            p_p_f = self.city_location_dict.get((x['city'],x['Standard_location']), city_average)
            return p_p_f*x['Area']
        
        result = copy_data.apply(func = calculare_price, axis = 1)

        return result.values