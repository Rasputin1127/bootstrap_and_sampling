import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


# Always make it pretty.
plt.style.use('ggplot')
font = {'weight': 'bold',
        'size':   16}
plt.rc('font', **font)

# data = stats.norm(0.1, 1.0)
# data = np.random.random_integers(0,100,100)

def bootstrap(x, resamples=10000):
    """Draw bootstrap resamples from the array x.

    Parameters
    ----------
    x: np.array, shape (n, )
      The data to draw the bootstrap samples from.
    
    resamples: int
      The number of bootstrap samples to draw from x.
    
    Returns
    -------
    bootstrap_samples: np.array, shape (resamples, n)
      The bootsrap resamples from x.
    """
    lst = []
    for i in range(0, resamples + 1):
        bootstrap_sample = np.random.choice(x, size=len(x), replace=True)
        lst.append(bootstrap_sample)
    return lst

def bootstrap_ci(sample, stat_function=np.mean, resamples=1000, ci=95):
    bootstraps = bootstrap(sample, resamples)
    boot_stats = []
    for i in bootstraps:
        boot_stats.append(stat_function(i))
    
    interval = (100-ci)/2
    print(interval)
    stat_value = stat_function(boot_stats)
    left_endpoint = np.percentile(boot_stats, interval)
    right_endpoint = np.percentile(boot_stats, 100-interval)

    # fig, ax = plt.subplots(1, figsize=(10, 4))
    # ax.hist(boot_stats, bins=50, density=True, color="black", alpha=0.5)
    # ax.set_title("boostrap sample means", fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=15)
    # plt.show()

    print("Sample Mean: {:2.2f}".format(stat_value))
    print("Bootstrap Confidence Interval for Population Mean: [{:2.2f}, {:2.2f}]".format(left_endpoint, right_endpoint))


# df = pd.read_csv("data/productivity.txt", header=None)
# print(df)
# arr = np.array(df).reshape(-1)
# print(arr)
# bootstrap_ci(arr, np.mean, 10000, 90)
# Output
# Sample Mean: 5.03
# Bootstrap Confidence Interval for Population Mean: [-0.23, 10.35]


# Based on the bootstrap confidence interval, what conclusions can you draw? What about if a 90% confidence interval were used instead?
# Suppose there are 100 programmers in the company. The cost of changing a monitor is $500 and the increase of one unit of productivity
# is worth $2,000, would you recommend switching the monitors? State the assumptions you are making and show your work.

# Based on the bootstrap confidence interval, it looks like the company will likely experience an overall improvement in productivity
# if we switch to Apple monitors. This improvement will reliably be larger than zero units of productivity, but there is a small chance
# that productivity will decrease slightly. With a 90 percent confidence interval, this moves in an even more positive direction, so even though we would
# be less confident in the results, there is an overall trend in the positive direction, and the mean for our confidence interval is positive
# both on the low and high ends. This means that we have 90% confidence that the company will experience an overall boost to productivity from this move.

# Since we are almost certain that we are gaining, on average, one unit of productivity per monitor change, then this is a positive
# move for the company. If the average gain is one unit ($2000), then that's a profit per programmer of $1500(2000-500). Even if
# our average turns out to only be half a unit, there is a profit overall. In fact, so long as our average increase in productivity
# stays above 0.25 units (2000*0.25 = 500), then it is overall profitable for the company, and we can be 95% confident that our
# average productivity will increase by more than that.

df1,df2 = pd.read_csv("data/law_sample.txt", usecols=[0,1])
print(df)
print(df2)

# corr = stats.pearsonr(arr)
# print(corr)
