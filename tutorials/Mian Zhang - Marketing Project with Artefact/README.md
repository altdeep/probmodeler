# CS7290 Applied Bayesian Machine Learning Course Project

Author: Mian Zhang

Project video link: https://youtu.be/lTjBh_s-r9Q

This is a real-world project created by Artefact company. Me(Mian Zhang), Sameer Marath, Ruthvik Ravindra, Biswaroop Maiti together joined this project, but we were assgined by seperated, individual tasks according to Artefact system. So we decided to make presentations and submit our work individually.

## Introduction

Artefact is an European digital marketing company, and they are doing a real-world project on attribution problem in marketing. To be more specific, their goal is to find the impact of various marketing channels on the sales of a specific product. In this way, their client will know which channel is worthwhile to spend money, since it can bring more profits. However, this goal will not be directly achieved in my work, because like I mentioned above, we were assigned by different tasks, and my jobs are some components of achieving this final goal.

## Code and Data

Artefact staff have already created a completed workflow, from DAG creation to model building, so my responsibility is to improve their work through tasks assigned to me, which means that some part of the code shown here are directly from Artefact. 

Since this is real-world project, each of us signed a Non-disclosure Agreement with Artefact, which means that we are not allowed to show any sensitive stuff, including but not limited to real data, brand names, product names, etc. Therefore, original dataset will not be uploaded. Also, as it can be seen from the Jupyter Notebook, there will no be code like ' df = pd.read_csv('XXXX.csv') ' or 'df.head()', in order to avoid presenting any potentially confidential information. However, I will still provide some fake datasets, for those who would like to try my work. But I cannot guarantee that using the fake data can obtain the same result, because some parts of Artefact's model is building on the dataset.

Now I'll give a brief introduction of all each file that I've uploaded:

(Artefact)Mian Zhang - CS7290 Project Code.ipynb ----> The Jupyter Notebook of the code implementing the project, from importing libraries to the final results. Analysis and explanation are also embedded. 

df_transformed.csv ----------------------------------> A fake dataset corresponding to the variable 'df_transformed' in the code

data_2.csv ------------------------------------------> A fake dataset corresponding to the variable 'data2' in the code

Project Code.pdf ---------------------------- -------> A pdf version copied from the Jupyter Notebook without any changes.

Presentation Slides.pdf -----------------------------> The slides for presentation.

If you want to view the project, I HIGHLY recommend you viewing in the following order:

1. If you just want to know what I did, view the Presentation Slides.pdf.
2. If you want to play with the code, after finish viewing the Presentation Slides.pdf, open (Artefact)Mian Zhang - CS7290 Project Code.ipynb and read data from 'df_transformed.csv' and 'data_2.csv' and define them to be 'df_transformed' and 'data2' respectively, rather than running the code from the top and creating these two dataframes after data cleaning, feature engineering and data transformation. Again, I cannot guarantee that using the fake data can obtain the same result, because some parts of Artefact's model is building on the dataset. 

## Setup

In my opinion, nothing special need to be stated here, except the version of PyMC3 and Arviz. The version of PyMC3 is 3.11.2. Arviz package used here is a development version, whcih can be installed using 'pip install git+git://github.com/arviz-devs/arviz.git'.

## Tasks, results and insights.

Task 1: Artefact staff has already built a directed graph verified by business knowledge and statistical test, and there are two cycles in the graph. We know that the probabilistic model should be build based on a DAG. So how to remove cycles. The solution is Topological Sorting, which is a graph traversal in which each node v is visited only after all its dependencies are visited. As a result, the direction between online marketing to offline marketing was changed from 'online to offline' to 'offline to online', which makes sense according to domain knowledge.

Task 2: Artefact staff has already got some metrics to evaluate models, including RMSE, MAPE, r-hat. They want to know which metric is best for model evaluation. After comparison, WAIC and LOO are the best, since they try to estimate the predictive accuracy of the model on unseen data, and they reveal the loss of information by providing a trade-off between goodness of fit and model complexity in a Bayesian way, which is something that those common metrics cannot achieve.

Task 3: Artefact staff has already built a hierarchical bayesian model using self-defined mu and sigma for variables and weakly informative priors for interaction coefficients between variables, but it cannot perfectly simulate data, because from the prior predictive check it can be known that the range of simulated value of the model is much wider. So how to choose the right priors for variables and coefficents. In brief, two changed were made. First,  variables in the model were all defined as normal distributed, so change the distribution of variables in the model to their true distribution according to the dataset. Second, all coefficients were defined to be standard normally distributed, but after sampling from a model without seeing any data, I modified mu and sigma of some interaction coefficients according to the summary statistics. Finally, the simulated data from prior distribution of the modified model has less extremte values, andWAIC and LOO are lower than original model, which indicates that the modified model has better predictive ability than original model.

Besides, for interaction priors, if we pick those with | mean | > 0.3 and std < 0.6, then we will get certian insights from some correlations: Launch more promotion activities, especially on holidays. Don’t make the price too high.

Last but not the least, for ppc chart problem that was raised by Professor Ness during my presentation, I carefully checked the official documents and tutorial of PyMC3 and was not able to find a more appropriate function to do the posterior predictive check visualization. I've also mentioned this problem in one of the stand-ups, but I was not able to get any better ideas about it. So I remain the ppc plot here to be the same as the previous edition, which can be a future work for me and Artefact people.

## Acknowledgements

First of all, thanks to Professor Ness for teaching and guidance for the whole semester, also for the opportunity to work with Artefact. Bayesian Modeling is brand new topic for me and now I find it very interesting and meaningful. Second, thanks to Aleksandra, Sarath and Deepak from Artefact. The support and guidance that they gave me for the past two months really helped me a lot, and I hope my work could at least help to push the project go further. Third, thanks to Sameer, Ruthvik and Biswaroop for the help during the past two months. It was really interesting and beneficial to be your partners.

