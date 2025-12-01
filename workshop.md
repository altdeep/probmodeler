# **Workshop: Probabilistic Machine Learning**  
**Instructor:** [Robert Osazuwa Ness](https://www.robertosazuwaness.com)  
*Author of* [***Causal AI***](https://www.robertosazuwaness.com/causal-ai-book/)

## **Overview**

This full-day workshop introduces intermediate and advanced practitioners to modern probabilistic machine learning, with a strong emphasis on uniting **deep probabilistic modeling** with Bayesian modeling techniques.

We focus on practical modeling workflows that combine statistical rigor with flexible, neural network–based architectures, using **Pyro** and **PyMC**. The workshop can be configured to:

- Emphasize **Pyro** for deep probabilistic models, amortized inference, and close integration with PyTorch, **or**
- Emphasize **PyMC** for expressive Bayesian modeling with solid NUTS/HMC workflows and integration with external deep learning components.

Participants will learn how probabilistic models support reasoning under uncertainty, scientific inference, and interpretable decision-making — and how to integrate these methods into real-world machine learning pipelines and research codebases.

The workshop blends hands-on demonstrations with conceptual foundations, making it suitable for both research scientists and industry data scientists who want to use probabilistic modeling as a core tool in their work.

---

## **Learning Objectives**

By the end of this workshop, participants will be able to:

- Build and critique Bayesian models using hierarchical structures, latent variables, and domain knowledge.
- Select and apply inference algorithms such as NUTS, HMC, and stochastic variational inference (SVI).
- Compare and combine computational frameworks **Pyro** and **PyMC**, choosing the right tool for the job.
- Design and implement **deep probabilistic models** such as VAEs and normalizing flows.
- Use probabilistic models for representation learning, forecasting, anomaly detection, causal inference, and simulation-based science.
- Develop workflows for model criticism, sensitivity analysis, and communicating uncertainty.

---

## **Target Audience**

- Research scientists and ML engineers  
- Data scientists with experience in Bayesian statistics or ML  
- Researchers working on reasoning, uncertainty, generative modeling, or causal inference  
- Anyone using probabilistic programming in applied or academic settings

Attendees should be comfortable with Python, basic probability theory, and gradient-based model training.

---

## **Schedule (Full Day: 8:00–17:00)**

### **8:00–8:30 — Introduction & Motivation**
- What probabilistic modeling is *for*  
- Why data scientists and researchers need uncertainty  
- When to choose probabilistic methods vs. classical ML or pure deep learning  
- How deep probabilistic models bridge structured reasoning and neural networks  

---

### **8:30–9:45 — Bayesian Modeling Workflows**
- Model specification, prior elicitation, and prior push-forward checks  
- Posterior inference and posterior predictive checks  
- Workflow patterns from statistical and ML practice  
- Case Study: Bayesian model for structured prediction  
- How these workflows map onto **Pyro** and **PyMC**

---

### **9:45–10:00 — Break**

---

### **10:00–11:30 — Inference Algorithms in Practice**
- Hamiltonian Monte Carlo and NUTS  
- Variational inference: SVI, amortization, reparameterization tricks  
- Trade-offs between accuracy, speed, latent variables, and scalability  
- Comparative demonstrations:
  - Using **PyMC** for NUTS/HMC workflows  
  - Using **Pyro** for SVI and deep probabilistic models  

---

### **11:30–12:30 — Lunch Break**

---

### **12:30–13:45 — Hierarchical & Latent Variable Models**
- Partial pooling and hierarchical structure  
- Mixture models, clustering, and density modeling  
- Topic models, factor analysis, probabilistic embeddings  
- Case Study: Latent structure in behavioral or interaction data  
- Implementing hierarchical and latent-variable models in **Pyro** or **PyMC**

---

### **13:45–14:00 — Break**

---

### **14:00–15:30 — Deep Probabilistic Models**
- Variational Autoencoders (VAEs)  
- Normalizing flows and expressive densities  
- Probabilistic neural networks and amortized inference  
- Hybrid models: combining neural nets with structured latent variables  
- Implementation focus:
  - **Pyro** for deep generative models and SVI  
  - How **PyMC** can interoperate with external deep learning components  
- Example: representation learning and uncertainty-aware prediction with VAEs

---

### **15:30–15:45 — Break**

---

### **15:45–17:00 — Causal Modeling, Decision-Making & Practitioner’s Playbook**
- Bayesian graphical models and structural assumptions  
- Identifiability and uncertainty-aware causal inference  
- Simulation-based causal analysis and counterfactual reasoning  
- Using deep probabilistic models in causal or “world-modeling” workflows  
- Practitioner’s playbook:
  - Choosing between Pyro and PyMC in practice  
  - Communicating uncertainty to stakeholders  
  - Common pitfalls and anti-patterns  
  - Recommended reading and practice roadmap  
- Open Q&A

---

## **Software and Tools**

This workshop uses:

- **Pyro** for deep probabilistic models, amortized inference, and close integration with PyTorch.
- **PyMC** for expressive Bayesian modeling with robust NUTS/HMC workflows and flexible integration with external deep learning models.

All examples are provided as runnable notebooks.

---

## **Suggested Background Reading**

- *Statistical Rethinking* — Richard McElreath
- *Bayesian Data Analysis* — Gelman et al.  
- Official documentation for Pyro and PyMC  
- Selected research papers provided during the workshop

---

## **Instructor**

**Robert Osazuwa Ness** is a researcher specializing in causal AI, generative modeling, and probabilistic programming. He is the author of *Causal AI*, has worked as an AI research scientist in both industry and academia, and is a contributor to leading open-source projects in probabilistic modeling.

Learn more at:  
https://www.robertosazuwaness.com
