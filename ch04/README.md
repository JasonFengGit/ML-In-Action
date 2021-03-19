# Naive Bayes Classifier
### Based on Bayes' Therom and Bayesian Decision Theory
![Bayes' Therom](https://wikimedia.org/api/rest_v1/media/math/render/svg/87c061fe1c7430a5201eef3fa50f9d00eac78810)
<br/>
![Bayesian Decision Theory](https://www.byclb.com/TR/Tutorials/neural_networks/Ch_4_dosyalar/image004.gif)
<br/>
In classification problem, **A** is category and **B** is feature.<br/>
Here we want **P(A | B)**, and we manage to use **P(B | A)**, **P(A)**, **P(B)**(usually ignored) to find **P(A | B)**
<br/><br/>
We find a class **Ci** with greatest **P(Ci | testFeature)** to decide the class.

### Naive Bayes Assumption (why Naive Bayes is *naive*)
1. independence among the features.
2. every feature is equally important

### Calculation Details
1. **feature** is defined as a **vector** to allow vector calculation using `numpy`
2. **P(feature | Ci)** = **P(f1, f2, ..., fn | Ci)** = **P(f1 | Ci) * P(f2 | Ci) * ... * P(fn | Ci)**, the second **=** sign works because of assumption on independence
3. **P(f | Ci)** could be very small. To avoid underflow caused by multipling **P(f | Ci)** together, we can instead calculate **log(P(f | Ci))**: `p1 * p2 * p3 -> log(p1 * p2 * p3) = log(p1) + log(p2) + log(p3)`
