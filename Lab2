import numpy as np
import scipy.stats as stats

name, surname = 'Артем', 'Плохов'
a, b = len(name), len(surname)
m = a*b
s = a+b
c, d = 20, 30
trust_level = 0.95

def conf_interval(x, std, quantile, unknown=False, parameter='mean'):
    n = len(x)
    if parameter == 'mean':
        quantile = 1 - quantile
        if unknown:
            t = stats.t.ppf(1-quantile/2, n-1)
        else:
            t = stats.norm.ppf(1-quantile/2, loc = 0, scale = 1)
        l = x.mean() - t*std/n**0.5
        r = x.mean() + t*std/n**0.5
        return (l, r)
    t = (stats.chi2.ppf((1-quantile)/2, n-1),stats.chi2.ppf((1+quantile)/2, n-1))
    s = x.std(ddof=1)
    if parameter == 'var':
        l = ((n-1)*(s**2))/t[1]
        r = ((n-1)*(s**2))/t[0]
        return (l, r)
    elif parameter == 'std':
        l = (((n-1)**0.5)*s)/((t[1])**0.5)
        r = (((n-1)**0.5)*s)/((t[0])**0.5)
        return (l, r)
    else:
        return None

pg = stats.norm.cdf(d, m, s) - stats.norm.cdf(c, m, s)
print('Name: %s %s' % (name, surname))
print('a = %i, b = %i, \nexpectation = %i, std = %i' % (a, b, m, s))
print('Interval: [%s, %s]' % (str(c), str(d)))
print('Probability (general): %s' % str(pg))

x = np.random.normal(m, s, size = 100)
y = np.random.choice(x, size=15, replace=False)
z = np.random.choice(x, size=15, replace=False)

for i in enumerate((x, y, z)):
    p = stats.norm.cdf(d, i[1].mean(), i[1].std()) - stats.norm.cdf(c, i[1].mean(), i[1].std())
    print('Probability (%i): %f' % (i[0], p))
    print('P_general - P_%i = %f' % (i[0], pg-p))

for i in enumerate((y, z)):
    mean = i[1].mean()
    var = i[1].var(ddof=1)
    std = i[1].std(ddof=1)
    e_m = m - mean
    e_std = s - std
    print('Sample %i:' % ((i[0])+1))
    print('Mean: %f' % mean)
    print('Variance: %f' % var)
    print('Std: %f' % std)
    print('e_m: %f' % e_m)
    print('e_std: %f' % e_std)
    print('\n')

print('Trust level: %f' % trust_level)
print('Сonfidence intervals:')
print('Expectation (known std): %s' % str(conf_interval(x, s, trust_level, unknown=False, parameter = 'mean')))
print('Expectation (unknown std): %s' % str(conf_interval(x, s, trust_level, unknown=True, parameter = 'mean')))
print('Variance: %s' % str(conf_interval(x, s, trust_level, unknown=True, parameter = 'var')))
print('Std: %s' % str(conf_interval(x, s, trust_level, unknown=True, parameter = 'std')))
