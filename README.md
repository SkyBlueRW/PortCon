
# Portfolio Construction with Convex optimization


## Technical Details in Convex Optimization

Convex Optimization is the minimization of a convex function on a convex set. What makes a convex function special is that local information (gradient or so) can lead us to global opmima. 

$$
\begin{aligned}
f(y) - f(x) >= \bigtriangledown f(x)(y - x)
\end{aligned}
$$

A convex Problem can be written in the following **standard form**. For such an optimization to be convex optimization, it is required that $f_0(x)$ if convex and $h_i(x)$ is affine. 

$$
\begin{aligned}
\min_x {f_0(x)} \\ 
f_i(x) &<= 0, \quad i=1...m\\
h_i(x) &= 0, \quad i=1...p
\end{aligned}
$$

**Duality**

In general, duality is the other view on an optimization problem. It is about to find the max one among a set of lower bound functions (Lagrangian Function). As long as $\lambda > 0$ (dual feasibility), $g(\lambda, \upsilon) <= f_0(x)$ holds for any x. Duality is a good indicator in terms of how good the current solution we have (Is duality gap close enoungh to 0?)

$$
\begin{aligned}
L(x, \lambda, \upsilon) &= f_0(x) + \sum_i^m{\lambda_i f_i(x)} + \sum_i^p{\upsilon_i h_i(x)} \\
g(\lambda, \upsilon) &= inf_x L(x, \lambda, \upsilon)
\end{aligned}
$$

Here we are able to transform the convex minimization problem into its dual problem. It is the best lower bound of the primal. 

- Weak duality: $d^{\star} <= p^{\star}$
- Strong duality: $d^{\star} = p^{\star}$
- For convex optimization, the strong duality holds under the Slater Condition (strict inequality constraint)

$$
\begin{aligned}
\max_{\lambda, \upsilon} & {g(\lambda, \upsilon)} \\
\lambda & >=0
\end{aligned}
$$

**KKT Condition**

A Necessary condition for optimal solutions of convex problem.

1. Primal feasibility
2. Dual Feasibility ($\lambda >= 0$)
3. Complementary slackness ($\lambda_i f_i(x) = 0$). It make sure that $f_0(x^{\star}) = L(x^{\star}, \lambda^{\star}, \upsilon^{\star})$
4. Gradient of L with respect to x is 0. It make sure that $g(\lambda^{\star}, \upsilon^{\star}) = L(x^{\star}, \lambda^{\star}, \upsilon^{\star})$



## Reference

- Boyd & Vandenberghe: Convex Optimization
- Cornuejols, Pena & Tutuncu: Optimization Methods in Finance
- Kolm, Tutuncu, Fabozzi (2013): 60 Years of Portfolio Optimization: Practical challenges and current trends
- Boyd, Busseti, Nystrup, Speth(2017): Multi-Period Trading via Convex Optimization