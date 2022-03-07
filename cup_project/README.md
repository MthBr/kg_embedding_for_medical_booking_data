# Source Code

This folder-module stores source code (python, R etc) which serves multiple scenarios.
During data exploration and model training, we have to transform data for a particular purpose.
We have to use the same code to transfer data during online prediction as well. So it better separates code from notebook such that it serves a different purpose.
It contains the pipeline steps identified by [IBM](https://developer.ibm.com/articles/the-lightweight-ibm-cloud-garage-method-for-data-science/
)
Ideally, all the process should work on a CI/CD pipeline.

## What to install
X pip install modin[all]
- pip install pathlib
- pip install ampligraph
- pip install grakn-client
X pip install pandas-compat


# Data ETL
The folder/module contains functions for ETL, all following the same convention.
For now, March 2020, this step does not require Modin!
This step involves:
* accessing the data source
* transforming the data, so it can be easily worked with, thanks to some Data Cleansing code
* make the data available to downstream analytics processes
Data Cleansing: 
* Set memberships
* Cross-field validation
This is useful to tackle problems like:
- Irrelevant column names
- Outliers
- Duplicates
- Missing data
- Columns that have to be processed
- Unexpected data values
## Preprocessing
* Parsing data
* Concatenating data
* Merging data
* Converting data types
* Duplicates and missing values
* Asserts
## Data Tidying
Tidy data sets main concept is to arrange data in a way that each variable is a column and each observation (or case) is a row.
## Cleaning
By running...


# Initial Data Exploration
This folder contains functions that calculate statistics and generate visualization, all following the same convention.
statistics and visualization on Data Set to identify good columns for modelling, potential data quality issues and anticipate potential feature transformations necessary.
Identify quality issues (e.g. missing values, wrong measurements, …)
Assess feature quality – how relevant is a certain measurement (e.g. use correlation matrix)
Get an idea on the value distribution of your data using statistical measures and visualizations
Step 1: Load data
Step 2: Explore data
Step 3: In-depth statistical measures and visualization
NOTE: data exploration is done also with R language.

# Exploratory analysis
For the quick overview are used methods and attributes of a DataFrame:
```
df.head() # show first 5 rows
df.tail() # last 5 rows
df.columns # list all column names
df.shape # get number of rows and columns
df.info() # additional info about dataframe
df.describe() # statistical description, only for numeric values
df['col_name'].value_counts(dropna=False) # count unique values in a column
```
The output of at least one of these will give us first clues where we want to start our cleaning.
Another way to quickly check the data is by visualizing it. We use bar plots for discrete data counts and histogram for continuous.
```
df['col_name'].plot('hist')
df.boxplot(column='col_name1', by='col_name2')
```
Histogram and box plot can help to spot visually the outliers. The scatter plot shows relationship between 2 numeric variables.
```
df.plot(kind='scatter', x='col1', y='col2')
```
Visualizing data can bring some unexpected results as it gives you a perspective of what’s going on in your dataset. Kind of view from the top.

## Exploring the data set
To make the process easier, you can create a DataFrame with the names of the columns, data types, the first row’s values, and description from the data dictionary.
As you explore the features, you can pay attention to any column that:
is formatted poorly,
requires more data or a lot of pre-processing to turn into useful a feature, or
contains redundant information,
since these things can hurt your analysis if handled incorrectly.
You should also pay attention to data leakage, which can cause the model to overfit. This is because the model will be also learning from features that won’t be available when we’re using it to make predictions. We need to be sure our model is trained using only the data it would have at the point of a loan application.


# Feature generation
This folder contains functions that calculate various features, all following the same convention. To generate features, you'll interface with the `featurebot.py` script.
## Generating features
By running...
### For KG
defining the schema and selecting the subset of patients.
It is important to use and install Grakn, in particular on Doker, for testing, ad stability.
```
docker stats
```

To install Grakn on Docker
```
docker run --name grakn -d -v $(pwd)/db/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:latest
docker run -d -v $(pwd)/db160/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:1.6.0
```

To use Docker
```
docker ps
docker ps -a
docker ps -s
docker exec -ti heuristic_davinci bash -c '/grakn-core-all-linux/grakn server status'
docker run --name grakn -d -v $(pwd)/db/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:latest
docker exec -ti grakn bash -c '/grakn-core-all-linux/grakn server status'
```

While on localhost
```
./grakn server start
./grakn server stop
./grakn server status
```
To load shemas, and test:
```
./grakn console --keyspace cupa_1 --file ../../PATH/Dropbox/MODAL_dropbox/CUP/Grakn/GraknCup/schemas/cup-network-schema_test1.gql
./grakn console --keyspace cup_1
clean
confirm
```
### For DL
Slecting subset of patines (with no NaNs, etc..)
Creation of time windows

## Adding new features

Add features for `featurebot.py`:
*  Define a new feature to generate in `featurebot.features_to_generate()`.
*  Add feature generation code in a separate file `<feature_name>.py`
*  Start `featurebot` to populate the database with features.
<!---
## Note on feature loading
After features are created, you can start training models. `dataset.py` handles the loading logic. When specifying features for training in the configuration file, you are actually selecting tables and columns, the pipeline groups together columns in the same table so they get loaded in a single call to the database. The summer pipeline required you to add a custom loading method for every table, which is good for security reasons but bad for flexibility. 
Right now, the pipeline uses a function that returns another function to load any group of columns (see `generate_loader_for_table` function in `dataset.py`), but the function is incomplete and will only work for tables that have a `parcel_id and `inspection_date` column.
--->



# Model Definition and Training
TODO, look at code


# Model Train - Evaluation and Deploy

TODO, look at code




# References
* [Cleaning and Prepping Data with Python for Data Science — Best Practices and Helpful Packages](https://medium.com/@rrfd/cleaning-and-prepping-data-with-python-for-data-science-best-practices-and-helpful-packages-af1edfbe2a3)
* [The complete beginner’s guide to data cleaning and preprocessing](https://towardsdatascience.com/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d)
* [Cleaning and Preparing Data in Python](https://towardsdatascience.com/cleaning-and-preparing-data-in-python-494a9d51a878)
* [Data Cleaning and Preprocessing](https://medium.com/analytics-vidhya/data-cleaning-and-preprocessing-a4b751f4066f)
* [Exploratory analysis](https://towardsdatascience.com/cleaning-and-preparing-data-in-python-494a9d51a878)
* [Configuring Python Projects](https://hackersandslackers.com/simplify-your-python-projects-configuration)
* [Manage Data Science project](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600)
* [Why and How to make a Requirements.txt](https://medium.com/@boscacci/why-and-how-to-make-a-requirements-txt-f329c685181e)
* [Data Science Project Folder Structure](https://dzone.com/articles/data-science-project-folder-structure)
* [Pipelines and Project Workflow](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow)
* [Structure Your Data Science Projects](https://towardsdatascience.com/structure-your-data-science-projects-6c6c8653c16a)








