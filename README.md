Generating digits from the MNIST dataset through stochastic differential equations.

This specific implementation uses the forward stochastic differential equation
$$
dx =-\frac{1}{2}\beta(t)\textbf{x}dt+\sqrt{\beta(t)}d\textbf{W} \quad t\in[0,1]
$$
Which has the corresponding solution, which we use to add noise to clean data points
$$
x_t=\exp\bigr[{-\frac{1}{2}\int_0^t\beta(s)ds}\bigl]x_0+\sqrt{1-\exp\bigr[{-\int_0^t\beta(s)ds}\bigl]}\space z \quad z \sim \mathcal{N}(0,I)
$$
We estimate the score using the equality
$$
\nabla_x \log p(x_t) = \mathbb{E}_{p(x_0|x_t)}\bigl[\nabla_x\log p(x_t|x_0)\bigr]
$$
