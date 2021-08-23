# Algorithmic fairness in deep generative image models
## Authors

* [**Vartika Tewari**](https://www.linkedin.com/in/vartika-tewari1992/)
* **Sai Srikanth Lakkimsetty**
* **Ritwik Anand**



The following videos give an overview of our project and goals and a code walkthrough: https://youtu.be/GfQtKxPfAOU

## Problem
To replicate the Deep Structural Causal Models for Tractable Counterfactual Inference[1]paper , and apply it to google cartoon faces dataset[3] and answer counterfactual queries on the same. 

We aim to explicitly model causal relationships with a fully specified causal models with no unobserved confounding and inferring exogenous noise via  normalising flows.

Our goal is to validate our causal assumptions; if our causal assumptions are valid, these simulations should align with our imagination.



## How to run the project
We have designed the project as a package for easy usage.

 `import causalfairness`


Also, you can run the jupyter notebook for a tutorial which is present in notebook folder.


### Prerequisites 

Use pip install package-name to install the dependencies. 

```
pyro
torch
numpy
pandas 

```

### License
This project is licensed under MIT - see the LICENSE.md file for details




## References
1. [Pawlowski, N., Castro, D. C., & Glocker, B. (2020). Deep structural causal models for tractable counterfactual inference. arXiv preprint arXiv:2006.06485.](https://arxiv.org/pdf/2006.06485.pdf)

2. [Normalizing Flows - Introduction (Part 1) — Pyro Tutorials 1.7.0 documentation](https://pyro.ai/examples/normalizing_flows_i.html)

3. [Cartoon Dataset](https://google.github.io/cartoonset/)

4. [Normalizing Flows for image modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html#Tutorial-11:-Normalizing-Flows-for-image-modeling)

5. [Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2017). “Density estimation using Real NVP,” In: 5th International Conference on Learning Representations, ICLR 2017.](https://arxiv.org/abs/1605.08803)
6. [Ho, J., Chen, X., Srinivas, A., Duan, Y., and Abbeel, P. (2019). “Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,” in Proceedings of the 36th International Conference on Machine Learning, vol. 97, pp. 2722–2730](https://arxiv.org/pdf/1902.00275.pdf)
7.[Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
