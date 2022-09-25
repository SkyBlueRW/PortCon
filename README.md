
# Portfolio Construction with Convex optimization


## Basic Technical Details in Convex Optimization

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
2. Dual feasibility : $\lambda >=0$
3. Complementary slackness $\lambda_i f_i(x) = 0$. It make sure that $f_0(x^{\star}) = L(x^{\star}, \lambda^{\star}, \upsilon^{\star})$
4. Gradient of L with respect to x is 0. It make sure that $g(\lambda^{\star}, \upsilon^{\star}) = L(x^{\star}, \lambda^{\star}, \upsilon^{\star})$

**Shadow Price**

Given the primal Problem

$$
\begin{aligned}
\min_x {f_0(x)} \\ 
f_i(x) &<= u_i, \quad i=1...m\\
h_i(x) &= v_i, \quad i=1...p
\end{aligned}
$$

The Associated Dual problem is 

$$
\begin{aligned}
\max_{\lambda, \upsilon} &{g(\lambda, \upsilon)} - u^T\lambda - v^T\upsilon\\
\lambda &>=0
\end{aligned}
$$

When Strong duality holds, The optimal value of dual problem equals the optimal value of primal problem. Hence the primal optimal value can be regared as a function of u and v: $p^{\star}(u, v)$. 

$$
\begin{aligned}
\lambda_i^{\star} = - \frac{\partial p^{\star}(0,0)}{\partial u_i} \\
\upsilon_i^{\star} = - \frac{\partial p^{\star}(0,0)}{\partial v_i} 
\end{aligned}
$$

## Literature Summary

1. Optimization Theory
    - Boyd & Vandenberghe: Convex Optimization
    - Cornuejols, Pena & Tutuncu: Optimization Methods in Finance
    - Kolm, Tutuncu, Fabozzi (2013): 60 Years of portfolio optimization: Practical challenges and current trends

2. Survey paper on Portfolio Optimization

    - Steinbach(2001); Rubinstein(2002); Fabozzi, Kolm, **Pachamanova & Focardi(2007)**; Markowitz(2014) provides great survey paper on the topic.

3. The Error Maximization Issue
    - DeMiguel, Garlappi & Uppal (2009), Jobson & Korkie (1981) and Jorion (1985) finds that the equal weight portfolio outperforms the mean variance efficient portfolio in a lot of scenarios
    - Green & Hollifield (1992) Find that Mean variance optimization leads to result that are not efficiently diversified.
    - Michaud (1998) argues that the navive optimization is an "Error Maximizers"
    - Kritzman (2006) argued that the error maximizer phenomenon is more significant when assets are highly correlated
    - 