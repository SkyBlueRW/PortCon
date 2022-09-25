
# Portfolio Construction with Convex optimization


## Technical Details in Convex Optimization

Convext Optimization is the minimization of a convext function within a convex set. What makes a convex function special is that local information (gradient or so) can lead us to global opmima. 

$f(y) - f(x) >= \bigtriangledown f(x)(y - x)$

A convex Problem can be written in the following **standard form**. For such an optimization to be convex optimization, it is required that $f_0(x)$ if convex and $h_i(x)$ is affine. 

$$
\begin{aligned}
\min_x {f_0(x)} \\ 
f_i(x) &<= 0, \quad i=1...m\\
h_i(x) &= 0, \quad i=1...p
\end{aligned}
$$

## Reference

- Boyd & Vandenberghe: Convex Optimization
- Cornuejols, Pena & Tutuncu: Optimization Methods in Finance
- Kolm, Tutuncu, Fabozzi (2013): 60 Years of Portfolio Optimization: Practical challenges and current trends
- Boyd, Busseti, Nystrup, Speth(2017): Multi-Period Trading via Convex Optimization