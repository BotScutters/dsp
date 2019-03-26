# [Think Stats Chapter 2 Exercise 4](http://greenteapress.com/thinkstats2/html/thinkstats2003.html#toc24) (Cohen's d):

Using the variable totalwgt_lb, investigate whether first babies are lighter or heavier than others. Compute Cohen’s d to quantify the difference between the groups. How does it compare to the difference in pregnancy length?

## Code

I used the code block below to investigate the question.

```python
# Compute and display summary statisticss for each group
display(firsts['totalwgt_lb'].describe(), 
        others['totalwgt_lb'].describe())

# Compute and display Cohen's effect size
print('Cohen\'s effect size: ', CohenEffectSize(firsts['totalwgt_lb'], others['totalwgt_lb']))
```

## Summary stats for weights of firstborn and other children

### Firstborn

```
count    4363.000000
mean        7.201094
std         1.420573
min         0.125000
25%         6.437500
50%         7.312500
75%         8.000000
max        15.437500
Name: totalwgt_lb, dtype: float64
```

### Other

```
count    4675.000000
mean        7.325856
std         1.394195
min         0.562500
25%         6.500000
50%         7.375000
75%         8.187500
max        14.000000
Name: totalwgt_lb, dtype: float64
```

Just looking at the summary statistics, we can see that firstborn children are roughly 0.125 lbs lighter than their future siblings, on average. That's a pretty small number compared to means > 7 and standard deviations around 1.4. Quantifying that with Cohen's _d_:

## Cohen's effect size

```
Cohen's effect size:  -0.088672927072602
```

A _d_ of ~0.089 indicates that the effect size here is about three times a large as with pregnancy length. However, it remains true that this is a pretty small effect size. Cohen suggests that effect sizes smaller than 0.2 are trivial.





