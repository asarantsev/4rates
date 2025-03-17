import numpy as np
import pandas as pd
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.api import OLS

def get_residuals(regression, x, y):
    residuals = np.zeros(len(x))
    for i in range(len(residuals)):
        residuals[i] = y[i] - (regression.slope*x[i] + regression.intercept)
    return residuals

def threePlots(data, name):
    name1 = 'QQPlot of ' + name + ' vs Normal Law'
    qqplot(data, line = 's')
    plt.title(name1)
    plt.savefig(name + '-res-qq.png')
    plt.close()
    name2 = 'ACF of ' + name
    plot_acf(data)
    plt.title(name2)
    plt.savefig(name + '-res-acf.png')
    plt.close()
    name3 = 'ACF of |' + name + '|'
    plot_acf(abs(data))
    plt.title(name3)
    plt.savefig(name + '-abs-acf.png')
    plt.close()
    
def skewKurt(data, name):
    skew = round(scipy.stats.skew(data), 3)
    kurt = round(scipy.stats.kurtosis(data), 3)
    print('\nThe skewness and kurtosis of ' , name, ' are ', skew, ' and ', kurt, 'respectively\n')
    print('Shapiro-Wilk p = ', round(100*scipy.stats.shapiro(data)[1], 1), '%')
    print('Jarque-Bera p = ', round(100*scipy.stats.jarque_bera(data)[1], 1), '%')
    
def ACF(data, name):
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), name, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), name, '\n')
    
df = pd.read_excel("4rates.xlsx", sheet_name = 'data')
vol = df["Volatility"].values[1:]
keys = ['BAA', 'AAA', 'Long', 'Short']
data4 = df[keys]

for key in keys:
    plt.plot(range(1927, 2025), df[key], label = key)
    plt.xlabel('Years')
    plt.ylabel('Rates')
    plt.title('Rate Plot')
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left')
plt.savefig('allrates.png')
plt.close()

for key in keys:
    series = df[key].values
    Reg = scipy.stats.linregress(series[:-1], np.diff(series))
    print('\nRegression for rates', key, '\n\n', Reg)
    resid = get_residuals(Reg, series[:-1], np.diff(series))
    threePlots(resid, key + ' Resid')
    skewKurt(resid, key + ' Resid')
    threePlots(resid/vol, key + ' NResid')
    skewKurt(resid/vol, key + ' NResid')
    ACF(resid, key + ' Resid')
    ACF(resid/vol, key + ' NResid')
    
spreads = [df['BAA'] - df['AAA'], df['AAA'] - df['Long'], df['Long'] - df['Short'], df['BAA'] - df['Long'], df['BAA'] - df['Short'], df['AAA'] - df['Short']]
spreadkeys = ['BAA-AAA', 'AAA-Long', 'Long-Short', 'BAA-Long', 'BAA-Short', 'AAA-Short']

for k in range(6):
    series = spreads[k].values
    key = spreadkeys[k]
    Reg = scipy.stats.linregress(series[:-1], np.diff(series))
    print('\nRegression for spreads', key, '\n\n', Reg, '\n\n')
    resid = get_residuals(Reg, series[:-1], np.diff(series))
    threePlots(resid, key + ' Resid')
    skewKurt(resid, key + ' Resid')
    threePlots(resid/vol, key + ' NResid')
    skewKurt(resid/vol, key + ' NResid')
    ACF(resid, key + ' Resid')
    ACF(resid/vol, key + ' NResid')

print('Rates VAR')
model = VAR(data4)
results = model.fit(1)
print(results.summary())
results.plot_acorr()
plt.savefig('VAR-residuals-ACF.png')
plt.close()
plt.show()
resid = results.resid

for key in keys:
    threePlots(resid[key], key + ' VAR Rates Resid')
    skewKurt(resid[key], key + ' VAR Rates Resid')
    threePlots(resid[key]/vol, key + ' VAR Rates NResid')
    skewKurt(resid[key]/vol, key + ' VAR Rates NResid')
    ACF(resid[key], ' VAR Rates Resid')
    ACF(resid[key]/vol, key + ' VAR Rates NResid')

print('Spreads VAR')
S1 = df['BAA'].values - df['AAA'].values
S2 = df['BAA'].values - df['Long'].values
spreads = pd.DataFrame({'BAA-AAA': S1, 'BAA-Long': S2})
plt.plot(range(1927, 2025), S1, label = 'BAA-AAA')
plt.plot(range(1927, 2025), S2, label = 'BAA-Long')
plt.xlabel('Years')
plt.ylabel('Spreads')
plt.title('Spread Plot')
plt.legend(bbox_to_anchor=(0.45, 0.95), loc='upper left')
plt.savefig('spreads.png')
plt.close()
model = VAR(spreads)
results = model.fit(1)
print(results.summary())
results.plot_acorr()
plt.savefig('VAR-spreads-ACF.png')
plt.close()
plt.show()
resid = results.resid

for key in ['BAA-AAA', 'BAA-Long']:
    threePlots(resid[key], key + ' VAR Spreads Resid')
    skewKurt(resid[key], key + ' VAR Spreads Resid')
    threePlots(resid[key]/vol, key + ' VAR Spreads NResid')
    skewKurt(resid[key]/vol, key + ' VAR Spreads NResid')
    ACF(resid[key], ' VAR Spreads Resid')
    ACF(resid[key]/vol, key + ' VAR Spreads NResid')
   
DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'BAA-AAA' : S1[:-1]/vol, 'BAA-Long' : S2[:-1]/vol})
RegS1 = OLS(np.diff(S1)/vol, DFreg).fit()
RegS2 = OLS(np.diff(S2)/vol, DFreg).fit()
print('Full Regression for BAA-AAA')
print(RegS1.summary())
resS1 = RegS1.resid
skewKurt(resS1, 'BAA-AAA Full Reg')
ACF(resS1, 'BAA-AAA Full Reg')
threePlots(resS1, 'BAA-AAA Full Reg')
print('Full Regression for BAA-Long')
print(RegS2.summary())
resS2 = RegS2.resid
skewKurt(resS2, 'BAA-Long Full Reg')
threePlots(resS2, 'BAA-Long Full Reg')
ACF(resS2, 'BAA-Long Full Reg')

DFcut = pd.DataFrame({'const' : 1/vol, 'BAA-AAA' : S1[:-1]/vol, 'BAA-Long' : S2[:-1]/vol})
RegS1 = OLS(np.diff(S1)/vol, DFcut).fit()
RegS2 = OLS(np.diff(S2)/vol, DFcut).fit()
print('Cut Regression for BAA-AAA')
print(RegS1.summary())
resS1 = RegS1.resid
skewKurt(resS1, 'BAA-AAA Cut Reg')
ACF(resS1, 'BAA-AAA Cut Reg')
threePlots(resS1, 'BAA-AAA Cut Reg')
print('Cut Regression for BAA-Long')
print(RegS2.summary())
resS2 = RegS2.resid
skewKurt(resS2, 'BAA-Long Cut Reg')
threePlots(resS2, 'BAA-Long Cut Reg')
ACF(resS2, 'BAA-Long Cut Reg')