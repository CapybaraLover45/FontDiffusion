Generating digits from the MNIST dataset through stochastic differential equations.

This specific implementation uses the forward stochastic differential equation (DDPM)


$$dx =-\frac{1}{2}\beta(t)\textbf{x}dt+\sqrt{\beta(t)}d\textbf{W} \quad t\in[0,1]$$


Which has the corresponding solution, which we use to add noise to clean data points at random times t (add noise function)


$$x_t=\exp\bigr[{-\frac{1}{2}\int_0^t\beta(s)ds}\bigl]x_0+\sqrt{1-\exp\bigr[{-\int_0^t\beta(s)ds}\bigl]}\space z \quad z \sim \mathcal{N}(0,I)$$

We use a neural network $$s_\theta(x,t)$$ to estimate the score using the equality
```math
\nabla_x \log p(x_t) = \mathbb{E}_{p(x_0|x_t)}\bigl[\nabla_x\log p(x_t|x_0)\bigr] \approx s_\theta(x,t)
```
The conditional probability can be directly calculated from the solution to the SDE, resulting in the objective function (dsm_loss function)

```math
\arg\min_\theta \mathbb{E}_{x_0 \sim p_{data}}\mathbb{E}_{p(x_0|x_t))}\bigl[\frac{1}{2}\| s_\theta(x,t)-\bigl( \nabla_x \log p(x_t|x_0)\bigr) \|^2\bigr] =
```
```math
 \arg\min_\theta \mathbb{E}_{x_0 \sim p_{data}}\mathbb{E}_{p(x_0|x_t)}\bigl[\frac{1}{2}\| s_\theta(x,t)-\biggl( -\frac{z}{\sqrt{1-\exp\bigr[{-\int_0^t\beta(s)ds}\bigl]}}\biggr) \|^2\bigr]
```

Then we can sample using the discretized reverse stochastic differential equation (sampling function)

```math
x_{t-\Delta t} =x_{t}+ \bigl[ -\frac{1}{2}\beta(t)x-\beta(t)s_\theta(x,t)\bigr](-\Delta t) + \sqrt{\beta(t)\Delta t}\xi \quad \xi \sim \mathcal{N}(0,I)

```
