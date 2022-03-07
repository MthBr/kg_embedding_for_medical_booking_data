# Data folder

This folder stores all the data.
Folder for storing subset data for experiments.
It includes both raw data and processed data for temporary use.


It should contain the following details:

* Where the data come from,
* What scripts under the scripts/ directory transformed which files under raw/ into which files under processed/ and cleaned/, and
* Why each file under cleaned/ exists, with optional references to particular notebooks. (Optional, especially when things are still in flux.)


## raw
 Storing the raw result which is generated from the preparation folder code.
 My practice is storing a local subset copy rather than retrieving data from remote data store from time to time. It guarantees you have a static dataset for rest of action. Furthermore, we can isolate from data platform unstable issue and network latency issue.

## processed
To shorten model training time, it is a good idea to persist processed data. It should be generated from “processing” folder.

## intermediate
Folder for storing binary (json or other formats) file for local use.
Storing intermediate result in here only.
For the long term, it should be stored in the model repository separately.
Besides binary model, you should also store model metadata such as date, size of training data.

## models

Model building such as tackling classification problem.
It should not just include model training part but also an evaluation part.
On the other hand, we have to think about multiple models scenario.
A typical use case is an ensemble model such as combining Logistic Regression model and Neural Network model.


## log_files
Log files, from long run tets, and time performances, analysis.


## kg_schema
The folder containing essentially metadata, in particular schema of the dataset.

