# CS7290 Final Project

Author: Ian Dardik

I worked with Quay Dragon for this project.  Although we ended up creating different models and writing separate code, we discussed our work and created our final presentation together.  

## Introduction
My project is creating a Bayesian model for the number of fantasy football points that NFL players will score from rushing.  I have implemented both an pooled and partially pooled model in Pyro, and I have included an earlier version in PyMC3 as well.  

## Updates since the final presentation
Here is the check list I presented on the final presentation:
1. WAIC.  
I was not able to get this working in time so I simply used Mean Squared Error (MSE) for comparing models.  
1. Work on the Attempts -> Points part of the model.  
I ended up choosing a constant to serve as the "points-per-attempt" multiplier.  I chose 0.8 which seems to work well.  Ideally I would like to explore the points-per-attempt concept further, perhaps as a variable in the model.  Unfortunately I did not have time for this exploration.  
1. Variables need to converge.  
The variables in both models now have no divergences and behave better now.  In the hierarchical model, I replace Normal distributions with LogNormal distributions in most places which may a helping factor.  
1. Write a hierarchical model.  
My "main" model is now hierarchical, where all 32 NFL teams are partially pooled on the alpha/beta priors to the Beta distribution for percent attempts, and on the mu (mean) prior for the number of attempts per team.  The results do seem to be better than the pooled model.  
1. Potentially translate the code to Pyro.  
I did end up translating the pooled model to Pyro and the code does look cleaner.  I eventually created a partially pooled model as well in Pyro.  

## Data
Here is the sequence of steps I took to scrape and clean the data:
1. I manually copied and pasted the data from [here](https://www.pro-football-reference.com/years/2020/rushing.htm) using the "Share & Export" -> "Get table as CSV (for Excel)" option. 
1. I copied the past 20 years of data into a single file called [raw_data.csv](https://github.com/iandardik/CS7290_project/blob/master/raw_data.csv)
1. I then cleaned up a few random characters of raw_data.csv using the "sed" linux program (see the exact commands [here](https://github.com/iandardik/CS7290_project/blob/master/clean_commands.txt)).  
1. I then used a python script to add the "Next Year" columns, i.e. stats such as Attempts and Points that each player will score in the subsequent year.  Here is the [script](https://github.com/iandardik/CS7290_project/blob/master/kv_data.py).  This script outputs a file called [data.csv](https://github.com/iandardik/CS7290_project/blob/master/data.csv). 
1. All of my Jupyter notebooks use data.csv from my github repository and perform additional clean up to the data.  The additional clean up is covered in the notebooks themselves.  

I will also note that it was helpful to visualize the data in excel using [this file](https://github.com/iandardik/CS7290_project/blob/master/data_analysis.xlsx).  

## Models
I have included two Jupyter Notebooks.  Please read them in order; I have written them assuming the reader will read the Pooled Pyro notebook followed by the Partially Pooled Pyro notebook.  
1. [Pooled Pyro](cs7290_ff_pyro_flat.ipynb)
1. [Partially Pooled Pyro](cs7290_ff_pyro_hier.ipynb)

## Results
Models were judged based on the square root of Mean Squared Error (MSE).  The pooled version has a sqrt(MSE) around 70 while the partially pooled model has a sqrt(MSE) around 60.  

## Discussion
Without WAIC or cross-validation it is tough to make a good comparison of the models, however the sqrt(MSE) is around 10 points lower for the partially pooled model which indicates it may be the better model.  One clear advantage that the partially pooled model has over the pooled is that we can generate a points distribution by conditioning on players' teams; in the pooled model we create a ranking using E(P(points|team attempt %)), while in the partially pooled model we create a ranking using E(P(points|team attempt %,team)).  

Unfortunately, at a sqrt(MSE) of around 60 and 70, the error rate of the model is rather high.  However, this meets my expectations because predicting fantasy football points is tough.  The player ranking lists that the models generate look much like the player rankings that ESPN might generate before the start of the season, and these lists never do a *great* job at predicting player rankings either.  

## Conclusion
If someone were to use this model for their fantasy football draft then I would recommend using the partially pooled model.  Unfortunately, the model has limitted use because it does not consider receiving yards or--in the case of some leagues--points per reception (PPR).  If the model was extended then it may be useful on draft day; I may extend the model in the future for my own purposes.  In addition to the ranking, the standard deviation of the distribution for each player will help me understand the uncertainty of the model's prediction on draft day.  Creating this model has been fun yet challenging, and has given me a taste for the complexities of Bayesian programming as well as modeling fantasy football points.  
