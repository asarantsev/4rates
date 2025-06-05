import numpy
import scipy
from statsmodels.tsa.stattools import acf

NSIMS = 10**4
NDATA = 50
skewParam = 3
L1orig = []
skews = []
L1abs = []
kurts = []

for sim in range(NSIMS):
    noise = scipy.stats.skewnorm.rvs(skewParam, size = NDATA)
    skews.append(scipy.stats.skew(noise))
    kurts.append(scipy.stats.kurtosis(noise))
    L1orig.append(sum(abs(acf(noise, nlags = 5)[1:])))
    L1abs.append(sum(abs(acf(abs(noise), nlags = 5)[1:])))

print('95% skewness = ', numpy.percentile(abs(numpy.array(skews)), 95))
print('99% skewness = ', numpy.percentile(abs(numpy.array(skews)), 99))
print('95% kurtosis = ', numpy.percentile(kurts, 95))
print('99% kurtosis = ', numpy.percentile(kurts, 99))
print('95% L1 original = ', numpy.percentile(L1orig, 95))
print('99% L1 original = ', numpy.percentile(L1orig, 99))
print('95% L1 absolute = ', numpy.percentile(L1abs, 95))
print('99% L1 absolute = ', numpy.percentile(L1abs, 99))