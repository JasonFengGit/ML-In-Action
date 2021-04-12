# Support Vector Machine
> More notes on [SVM.pdf](https://github.com/JasonFengGit/ML-In-Action/blob/master/ch06/SVM.pdf)<br/>

## SVM idea
To find a line/hyperplane `w` that "best" seperate the data points<br/>
![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/350px-SVM_margin.png)

### Using Lagrange Multipliers
**Dual Form**<br/>
![](http://www.sciweavers.org/tex2img.php?eq=%7B%5Cdisplaystyle%20%7B%5Ctext%7Bmaximize%7D%7D%5C%2C%5C%2Cf%28%20%5Calpha%20%29%3D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%20%5Calpha_%7Bi%7D-%7B%5Cfrac%20%7B1%7D%7B2%7D%7D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%5Csum%20_%7Bj%3D1%7D%5E%7Bm%7Dy_%7Bi%7D%20%5Calpha_%7Bi%7Dy_%7Bj%7D%20%5Calpha_%7Bj%7DK_%7Bij%7D%2C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)<br/>
**Take Derivative to get**<br/>
![](http://www.sciweavers.org/tex2img.php?eq=%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%20%5Calpha_%7Bi%7Dy_%7Bi%7D%3D0&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)<br/>
![w=\sum _{i=1}^{m} \alpha_{i}y_{i}x_{i}](http://www.sciweavers.org/tex2img.php?eq=w%3D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%20%5Calpha_%7Bi%7Dy_%7Bi%7Dx_%7Bi%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)<br/>
**Bounds for alphas**<br/>
![](http://www.sciweavers.org/tex2img.php?eq=0%20%20%5Cleq%20%20%5Calpha%20%5Cleq%20C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

## Sequential Minimal Optimization
> [SMO Slides of NCTU](https://dsmilab.github.io/Yuh-Jye-Lee/assets/file/teaching/2017_machine_learning/SMO_algorithm.pdf)<br/>

**Idea**<br/> Instead of solving the big quadratic equation(dual form), update a **pair** of alphas each time (hopefully they converge in the end), while maintaining  ![](http://www.sciweavers.org/tex2img.php?eq=%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%20%5Calpha_%7Bi%7Dy_%7Bi%7D%3D0&bc=White&fc=Black&im=jpg&fs=10&ff=arev&edit=0)<br/>

**Update Function**<br/>
![](http://www.sciweavers.org/tex2img.php?eq=%20%5Calpha_j%27%3D%20clip%28%5Calpha_j%2B%20%5Cfrac%7By_j%28E_j-E_i%29%7D%7B2K_%7Bij%7D-K_%7Bii%7D-K_%7Bjj%7D%7D%20%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

### Kernels<br/>
use **Kernel functions** to map data points that are not linearly separable to higher dimensions.

**Radial bias function**<br/>
![](http://www.sciweavers.org/tex2img.php?eq=K%28x%2C%20y%29%3Dexp%28%20%5Cfrac%7B-%20%5C%7C%20x-y%20%5C%7C%5E%7B2%7D%7D%7B2%20%5Csigma%20%5E%7B2%7D%20%7D%20%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
