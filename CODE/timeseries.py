import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('agg_nasa.csv')

df_ca = df[(df['US_state'] == 'California')]

df_ca_grouped = df_ca.groupby('week')['frp'].mean().reset_index()

df_ca_grouped['week'] = pd.to_datetime(df_ca_grouped['week'] + '-1', format='%Y-%W-%w')

df_ca_grouped.set_index('week', inplace=True)

decomposition = seasonal_decompose(df_ca_grouped['frp'], model='additive', period=52)


plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Time Series Decomposition of Average FRP in California', fontsize=16)
plt.tight_layout()
plt.show()

