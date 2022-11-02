import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pmdarima import auto_arima
from pmdarima import arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# %config InlineBackend.figure_format = 'svg'
class Generator:
    def __init__(self, odata):
        self.df = pd.DataFrame(odata, columns=['target'])
        self.df.index = pd.util.testing.makeDateIndex(len(self.df), '30S')
        self.ts = self.df['target']
        self.predict = None
        self.result = None

    def train(self):
        order, sea_order = self.auto_parameters(self.ts, int(len(self.ts) * 2 / 3))
        model = sm.tsa.statespace.SARIMAX(self.ts, order=order, seasonal_order=sea_order)
        results = model.fit()
        self.result = results
        return results

    def get_forest(self, num):
        predict = self.result.forecast(num)
        self.predict = predict
        return predict

    def draw(self):
        self.predict.plot(color='green', label='Forecast')
        self.ts.plot(color='blue', label='Original')
        plt.show()

    def auto_parameters(self, data, s_num):
        kpss_diff = arima.ndiffs(data, alpha=0.05, test='kpss', max_d=s_num)
        adf_diff = arima.ndiffs(data, alpha=0.05, test='adf', max_d=s_num)
        d = max(kpss_diff, adf_diff)
        D = arima.nsdiffs(data, s_num)

        stepwise_model = auto_arima(data, start_p=0, start_q=0,
                                    max_p=6, max_q=4, max_d=2, m=s_num,
                                    seasonal=True, d=d, D=D, trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
        print("AIC: ", stepwise_model.aic())
        print(stepwise_model.order)  # (p,d,q)
        print(stepwise_model.seasonal_order)  # (P,D,Q,S)
        # print(stepwise_model.summary())  # 详细模型
        return stepwise_model.order, stepwise_model.seasonal_order
