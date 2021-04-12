# Support Vector Machine
> More notes on [SVM.pdf](https://github.com/JasonFengGit/ML-In-Action/blob/master/ch06/SVM.pdf)<br/>

## SVM idea
To find a line/hyperplane `w` that "best" seperate the data points<br/>
![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/350px-SVM_margin.png)

### Using Lagrange Multipliers
**Dual Form**<br/>
![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/1.jpg)<br/>
**Take Derivative to get**<br/>
![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/2.png)<br/>
![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/3.png)<br/>
**Bounds for alphas**<br/>
![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/4.png)

## Sequential Minimal Optimization
> [SMO Slides of NCTU](https://dsmilab.github.io/Yuh-Jye-Lee/assets/file/teaching/2017_machine_learning/SMO_algorithm.pdf)<br/>

**Idea**<br/> Instead of solving the big quadratic equation(dual form), update a **pair** of alphas each time (hopefully they converge in the end), while maintaining  ![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/2_smaller.png)<br/>

**Update Function**<br/>
![](https://bit.ly/3dXm8XR)

### Kernels<br/>
use **Kernel functions** to map data points that are not linearly separable to higher dimensions.

**Radial bias function**<br/>
![](https://raw.githubusercontent.com/JasonFengGit/ML-In-Action/master/ch06/formula_imgs/5.png)
