# Logistic Regression
### Based on Logistic Fuction Sigmoid
![Sigmoid](https://wikimedia.org/api/rest_v1/media/math/render/svg/a2ccf74b6142eee1c55895ba62531ba11871cf90)

### Optimization
Using **Gradient Ascent** to find the best weights:  **Weights** = **Weights** + **alpha**(learning rate) * **Directional Derivative**<br/>
Classify **X** with whether **sigmoid(W.transpose() * X)** > **0.5**

### Dealing with missing data
1. Use the feature’s mean value from all the available data. 
2. Fill in the unknown with a special value like -1.
3. Ignore the instance.
4. Use a mean value from similar items.
5. Use another machine learning algorithm to predict the value.

### Overflow when calculating sigmoid
use **0.5 + 0.5 * tanh(x/2)** instead.
