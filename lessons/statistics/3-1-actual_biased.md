# [Think Stats Chapter 3 Exercise 1](http://greenteapress.com/thinkstats2/html/thinkstats2004.html#toc31) (actual vs. biased)

Something like the class size paradox appears if you survey children and ask how many children are in their family. Families with many children are more likely to appear in your sample, and families with no children have no chance to be in the sample. Use the NSFG respondent variable NUMKDHH to construct the actual distribution for the number of children under 18 in the household. Now compute the biased distribution we would see if we surveyed the children
and asked them how many children under 18 (including themselves) are in their household. Plot the actual and biased distributions, and compute their means. As a starting place, you can use chap03ex.ipynb.

## Code

I used the code block below to investigate the question.

```python
# Read in data
resp = nsfg.ReadFemResp()

# Compute actual pmf and print mean
actual_pmf = thinkstats2.Pmf(resp['numkdhh'], label='actual')
print('Actual mean: ', actual_pmf.Mean())

# Define function to compute biased pmf
def bias_pmf(pmf, label):
    new_pmf = pmf.Copy(label=label)
    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
    new_pmf.Normalize()
    return new_pmf

# Compute biased pmf and print mean
biased_pmf = bias_pmf(actual_pmf, label='observed')    
print('Biased mean: ', biased_pmf.Mean())

# Plot actual and biased distributions
thinkplot.PrePlot(2)
thinkplot.Pmfs([actual_pmf, biased_pmf])
thinkplot.Show(xlabel='# of kids', ylabel='Probability')
```

## Results

```
Actual mean:  1.024205155043831
Biased mean:  2.403679100664282
```

![](/3-1.png)

