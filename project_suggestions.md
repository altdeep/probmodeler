# Project Suggestions

## Create a Bayesian data pipeline

1. Fetches data from a public API.
1. Submits the data as an input to a probabilistic model.
1. Generates samples from a posterior.
1. Takes posterior samples and stores them in an S3 bucket.

The storage in the S3 bucket should be serialized such that you can query the samples using [Amazon Athena](https://aws.amazon.com/blogs/big-data/analyzing-data-in-s3-using-amazon-athena/).

## Implement a barebones probabilistic programming language

