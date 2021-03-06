# An Asymptotic Test for Conditional Independence using Analytic Kernel Embeddings
Code of the [paper](https://arxiv.org/pdf/2110.14868.pdf) by Meyer Scetbon, Laurent Meunier and Yaniv Romano.

## A Simple Test for Conditional Independence
In this work, we propose a new computationally efficient test for conditional independence based on the Lp distance between two kernel-based representatives of well suited distributions. By evaluating the difference of these two representatives at a finite set of locations, we derive a finite dimensional approximation of the Lp metric and obtain its asymptotic distribution under the null hypothesis of conditional independence which is simply the standard normal distribution. We then design an asymptotic statisical test from it and show that it outperforms state-of-the-art methods as it is the only test able of to control the type-I error while having high power. In the following figure, we compare the performances of SoTA tests with our method on two different models where either the random variables X and Y are independent conditonally on Z or not.
![figure](figures/comparison_test.png)

## On the Implementation of the Test
In our code, we allow the optimization of the hyperparameters involved in the Regularized Least-Squares estimators. To do so, we propose a Gaussian Process (GP) regression that maximizes the likelihood of the observations. While carrying out a precise GP regression can be prohibitive, in practice, we run this method only on a batch of size 200 observations randomly selected and we perform only 15 iterations for choosing the hyperparameters involved in the RLS problems. We consider the implementation of the GP proposed in [scikit-learn](https://github.com/scikit-learn/scikit-learn) where we constraint the maximum number of iterations of the GP. More precisely, in the file sklearn/gaussian_process/_gpr.py we have modified the function called "_constrained_optimization" by adding the following options to the optimizer: 
options={'disp': None, 'maxcor': 10, 'ftol': 1e-3, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15, 'iprint': - 1, 'maxls': 20}.

In this [file](https://github.com/meyerscetbon/lp-ci-test/blob/main/toy_examples.py) we provide some toy examples where we test our method on the syntetic datasets presented in the [paper](https://arxiv.org/pdf/2110.14868.pdf).
