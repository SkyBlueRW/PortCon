- [Portfolio Construction with Convex optimization](#portfolio-construction-with-convex-optimization)
  * [Available Models](#available-models)
    + [Mean Variance Optimization](#mean-variance-optimization)
    + [Risk Based Optimization](#risk-based-optimization)
    + [Black-Litterman](#black-litterman)
  * [Constraint Attribution](#constraint-attribution)
  * [Basic Technical Details in Convex Optimization](#basic-technical-details-in-convex-optimization)
  * [Reference](#reference)

# Portfolio Construction with Convex optimization


## Available Models

### Mean Variance Optimization

```python

from mosek_api import mean_variance, max_ret, max_ir

optimized_portfolio = mean_variance(
    alpha, risk_aversion, cov, active_risk=False, active_alpha=False,
    long_only=True, return_full_result=False,
    benchmark=None, budget=None, max_budget=None,
    max_holding=None, min_holding=None, max_active_holding=None, min_active_holding=None,
    industry=None, max_ind=None, min_ind=None, max_active_ind=None, min_active_ind=None,
    style=None, max_style=None, min_style=None, max_active_style=None, min_active_style=None,
    prev_holding=None, max_turnover=None
    )


optimized_portfolio = max_ret(
    alpha, active_alpha=False, long_only=True, return_full_result=False,
    benchmark=None, budget=None, max_budget=None,
    cov=None, max_std=None, max_active_std=None,
    max_holding=None, min_holding=None, max_active_holding=None, min_active_holding=None,
    industry=None, max_ind=None, min_ind=None, max_active_ind=None, min_active_ind=None,
    style=None, max_style=None, min_style=None, max_active_style=None, min_active_style=None,
    prev_holding=None, max_turnover=None
    )

optimized_portfolio = max_ir(
    alpha, cov, max_holding=None, long_only=True
    )
```

### Risk Based Optimization 

Traditional Mean-Variance optimization is **notoriously sensitive** to errors in the estimation of inputs. One way that can help to mitigate the gap here is to base optimization only on risk related inputs. Risk based optimization is legit at least from the following two pespectives. From a statistical perspective, the estimation of risk measures (covariance, variance, etc..) is usually more robust than that of expected return. From a financial theory pespective, efficient exposure to risk is the key to harvest risk premium. One can also take the risk based optimization as a mean variance optimization with strong structure assumed as follows:

1. Min Variance: 
    
    a. Assmumption to equal MVO:When equal expected return for all securities
    
    b. Optimization condition

$$
\begin{aligned}
\frac{\partial{\sigma (w_i)}}{\partial w_i} = \frac{\partial{\sigma (w_j)}}{\partial w_j}
\end{aligned}
$$

2. Risk Parity: 
    
    a. Assumption to equal MVO: When equal return to risk contribution for all securities
    
    b. Optimization condition

$$
\frac{\partial{\sigma (w_i)}}{\partial w_i} * w_i = \frac{\partial{\sigma (w_j)}}{\partial w_j} * w_j
$$       


3. Risk Budget:

    b. Optimization condition

$$
\frac{\partial{\sigma (w_i)}}{\partial w_i} * w_i / b_i = \frac{\partial{\sigma (w_j)}}{\partial w_j} * w_j / b_j
$$  


4. Maximum Diversification: 
    
    a. Assumption to equal MVO: equal correlation and equal expected return for all securities
    
    b. Optimization condition

$$
\frac{\partial{\sigma (w_i)}}{\partial w_i} * \frac{1}{\sigma_i} = \frac{\partial{\sigma (w_j)}}{\partial w_j} * \frac{1}{\sigma_j}
$$  


**Min Variance**

```python
from mosek_api import risk_budget

optimized_weight = min_var(cov, long_only=True, budget=1, max_holding=None, min_holding=None)
```


**Risk Budget**


```python
# Code Demo

from mosek_api import risk_budget

optimized_weight = risk_budget(target_budget, covariance_matrix)
```

When the risk measure is a first order homogeneous function of portfolio, we will be able to decompose the total portfolio risk $R_w$ as summations risk contribution $rc_{w}$ for each securitys.

$$
\begin{aligned}
R_w &= \sum_{i=1}^N rc_{w, i}\\
&= \sum_{i=1}^N w_i \frac{\partial R_w}{\partial w_i}
\end{aligned}
$$

Based on the additivity of risk contribution. We can allocate predefined risk (risk contribution): $b_i$ to each securities.

$$
\begin{aligned}
w_i * \frac{\partial R_w}{\partial x_i} &= b_i * R_w
\end{aligned}
$$

The above condition is unfortunately not convex directly. While when variance is used as risk measure, we will be able to formulate a restricted version of convex optimization on it. The first order optimal condition of the following optimization problem is exactly the risk budget condition. Hence we can rely on it for risk budget optimization. While it should be noted that, with such implementation, we wil not be able to add further constraints since it will change the 1st order condition.

$$
\begin{aligned}
Min_w \frac{1}{2} w^T \Sigma w - c b^T log(z^T w) \\
z^T w >=0
\end{aligned}
$$

variable c can be used to scale the summation of budget b. Since leverage does not change the risk budget for first order homogenours risk measures. We can scale w to summation of 1 for fully invest constraint. 


**Risk Parity**

Risk parity portfolio is the equal risk contribution case. It is applied with equal risk contribution with the risk budget optimizer.

```python
# Code Demo

from mosek_api import risk_parity

optimized_weight = risk_parity(covariance_matrix)
```

**Maximum Diversification Portfolio**

Maximum Diversification portfolio as suggested by its name seeks to maximize the diversification ratio $\frac{w^T \sigma}{\sqrt{w^T \Sigma w}}$. In the financial theory, diversification has long been identified as the only "free lunch" in the investment world. The maximum diversification portfolio seeks to exploit the full potential of diversification in terms of correlation structure. It should be noted that the optimized portfolio is rather sensitive to estimation of correlation, which might cause problem in crash periods when correlation structure broke.

The maximum diversification is applied with quadractic optimization hence it can be applied with customized feasible set (other constraints like standard deviation threshold, max holding, etc)

```python
from mosek_api import max_div

# Pure Max diversification with fully invest 
optimized_weight = max_div(cov, long_only=True)

# Max diversification with vol target
optimized_weight = max_div_vol(cov, max_std,
                max_holding=None, min_holding=None,
                budget=1, long_only=True)
```

### Black-Litterman

Black-Litterman is a Bayesian based method for robust portfolio.


## Constraint Attribution

```python
from mosek_api import implied_alpha_decompose, holding_decompose, return_attribute

constraint_attr = holding_decompose(res_dict)
constraint_attr = return_attribute(ret, holding_dict, block_name=None)
```

The goal of constraint attribution is to decompose the ex-post performance into each constraints applied in the optimization. In practical, investors usually add constraint on the portfolio for robust ex-post performance. From a ex-ante perspective, the optimized portfolio will always have worse characteristic since the feasible set is smaller. It requires another perspective to look into how is the constraints working for the portfolios. Constraint attribution is one of the techniques that we can rely on for such purpose.

**Lagrangian Dual Decomposition**

For a portfolio optimization problem as below. Function f can stand for transaction cost, market impact ..etc. Function g refers to portfolio constraint that we look to decompose. 

$$
\begin{aligned}
\max_x \quad &{\alpha^Tx - \frac{1}{2} \lambda x^TQx + \sum f_i(x)} \\
g_i(x) &<=0
\end{aligned}
$$

Assuming all constraint function g are diffrentiable, Scherer & Xu (2007) provide the following decomposition 

**First Way to Decompose**

$$
\begin{aligned}
\lambda Qx^{\star} &= \alpha + \sum \bigtriangledown f_j(x^{\star}) - \sum \pi_i \bigtriangledown g_i(x^{\star}) \\
x^{\star} &= \frac{1}{\lambda} Q^{-1}\alpha + \sum \frac{1}{\lambda} Q^{-1} \bigtriangledown f_j(x^{\star}) - \sum \pi_i  \frac{1}{\lambda} Q^{-1}\bigtriangledown g_i(x^{\star}) \\
x^{\star} &= x_u + \sum_j x_j + \sum_i x_i
\end{aligned}
$$

- The above decomposition originates from the first order condition of KKT. $\pi_{i}$ is the shadow price of constraint $g_i$
- Formula 1 above is the decompostiohn of implied alpha. Formula 2 is the active holding decomposition
- $\lambda Q x^{\star}$ is called as implied alpha due to the reason that in an unconstraint MVO optimization, it is the alpha that will lead to $x^{\star}$ as optimal portfolio
- $x_u = \frac{1}{\lambda} Q^{-1} \alpha$ is the solution of the unconstrained optimization $\max_x{\alpha^T x - \frac{1}{2}\lambda x^TQx}$
- $x_u$ is actually a multiple of the characteristic portfolio of alpha; $\frac{Q^{-1} \alpha}{\alpha^T Q^{-1} \alpha}$ (The minimum variance portfolio given exposure to alpha as 1). 
- When dealing with linear constraint $Ax <= 0$, $\sum \pi_i  \frac{1}{\lambda} Q^{-1}\bigtriangledown g_i(x^{\star}) = \sum \pi_i \frac{1}{\lambda} Q^{-1}A_i$, $A_i$ is the i-th constraint of all constraints. We can interprete it as the **shadow price weighted characteristic portfolio of each constraints**. When the constraint is not binding, the shadow price will be 0 and the constraint will not play with any rule in the decompostion
- Tutuncu (2012) argued that, the decomposition of holding is not intuitive due to the high correlation between different constraints (the existence of Q in the equation). In such case the decomposition of implied alpha can provide a more precise decomposition.

**Second Way to Decompose**

$$
\begin{aligned}
x^{\star} &=  \beta_{cu} x_u + x_{orthogonal} \\
x_{u} &= \frac{1}{\lambda} Q^{-1} \alpha \\
x_{orthogonal} &= \sum (\beta_{tu}x_u - \frac{1}{\lambda}Q^{-1} \bigtriangledown f_j(x^{\star})) + \sum \pi_i(\beta_{pu} x_u - \frac{1}{\lambda}Q^{-1} \bigtriangledown g_i(x^{\star})) \\
\beta_{cu} &= \frac{x^{\star}Qx_u}{x_u^TQx_u} \\
\end{aligned}
$$

- Another way to further decompose is to project the wegiht on to alpha charateristic portfolio and those orthogonal
- Orthogonal part can be taken as unrewarded risk 

**Literature on Constraint Attribution**

- Grinold (2005) is the first  to propose the use of Lagrangian Dual decomposition (Basd on the first order condition of KKT). to addres the attribution analysis of portfolio constraints.
- Tutuncu (2012) provides a sumarization on previous research on the topics. He pointed that for a Mean Variance optimization with linear constraint, we can use the Lagrangian dual decomposition to perform the following decomposition on the optimized portfolio:
    
        a. Utility decomposition

        b. Implied alpha decomposition

        c. Active weight decomposition

        d. Alpha prediction return decomposition

- Scherer & Xu (2007) improved the method based on Grinold (2005). They propose to perform Lagrangian Dual decomposition on the quadratic utility of a MVO. They believed that the investors cared about the utility function instead of weights on individual securities.
- Lee & Stefek (2009) provides further analysis in terms of ex-ante analysis. The optimized portfolio with constraint can be  decomposed into the optimal portfolio (without constraint) and characteristic portfolios weighted by their shadow price. They further decomposed the characteristic portfolio into the projection on alpha characteristic portfolio and those orthogonal to the alpha characteristic portfolio. The second part can be viewed as the ex-ante risk that are note rewarded with ex-ante return
    
        Optimized Portfolio (with constraint)

            a. Optimal portfolio (without constraint)
            b. Corresponding characteristic portfolio for each constraint (Weighted by shadow price)

                I. Projection on alpha characteristic portfolio
                II. Residual (Orthogonal to alpha characteristic portfolio)   

- Stubbs& Vandenbussche (2010) focused more on the ex-post return attribution. Using the ex-post return to understand how is constraint impacting the portfolio. It can help to answer the question: can a certain optimization constraint help to improve ex-post performance? They also provide ways to do such attribution for risk constained portfolio. It will provide a slightly different economic intuition for the decomposition.




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

## Reference

1. Optimization Theory
    - Boyd & Vandenberghe: Convex Optimization
    - Cornuejols, Pena & Tutuncu: Optimization Methods in Finance

2. Survey paper on Portfolio Optimization

    - Steinbach(2001); Rubinstein(2002); Fabozzi, Kolm, **Pachamanova & Focardi(2007)**; Markowitz(2014) provides great survey paper on the topic.
    - Kolm, Tutuncu, Fabozzi (2013): 60 Years of portfolio optimization: Practical challenges and current trends

3. Constraint Attribution
    - Bernd Scherer & Xiaodong Xu (2007): The Impact of Constraints on Value-Added
    - Robert Stubbs & Dieter Vendenbussche (2010): Constraint Attribution
    - Jennifer Bender, Jyh-Huei Lee & Dan Stefek (2009): Decomposint the Impact of Portfolio Constraints

4. Risk Based Optimization
    - Clarke, De Silva & Thorley (2013): Risk Parity, Maximum Diversification, and Minimum Variance: An Analytic Perspective.
    - Choueifaty & Coignard (2008): Toward Maximum Diversification