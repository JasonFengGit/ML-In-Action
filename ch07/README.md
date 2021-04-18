# AdaBoost Meta-algorithm & Classification Imbalance
> [MIT's Boosting Notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/readings/MIT6_034F10_boosting.pdf)<br/>
> [Cornell's Boosting Note's](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote19.html)


## AdaBoost

### General Idea
- Use the **power of a crowd of weighted experts**(weak classifiers).
- Each expert might have their own specialized space on the sample space, and we want a way to train them such that their **"wrong portions"** do not overlap, so when voting, the "weighted majority" always get the correct answer.
- We also weight the samples and **update** them each time such that samples being misclassified previously is weighted more and vise versa. 

### Some Crucial Formulas
<img src="https://render.githubusercontent.com/render/math?math=\epsilon=\sum_{wrong}W_i" height=30 alt>
<img src="https://render.githubusercontent.com/render/math?math=\alpha=\frac{1}{2}ln(\frac{1-\epsilon}{\epsilon})" height=30 alt>
<img src="https://render.githubusercontent.com/render/math?math=W^{updated}=W^{old}e^{-\alpha*H(x)y(x)}" height=24 alt>

### When the Math Sings
It turns out that the **sum of updated weights** of **correctly classified** and **incorreclty ones** both equals to **1/2** after applying the formulas above. As a result, we can **easily** find the **updated weights** by **scaling** the weights of the misclassified samples up to **1/2** and by **scaling** the weights of the correctly classified samples down to **1/2**.
> more calculations in detail on MIT's Boosting Notes

## Classification Imbalance
### Problem
1. num of positive examples != num of negative examples
2. the cost for misclassification are different from positive and negative examples(spam example)

### What we can do
1. use an alternative performance metrics: precision, recall, and ROC
2. use a cost function to address different misclassification costs
3. undersampling and oversampling(CAUTION: this practice can lead to overfitting)
