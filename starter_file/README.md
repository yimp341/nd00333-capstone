# Student alcohol consuption 

In this project, we aim to predict the alcohol consumption among students, by means of both AutoML and Hyperparameter tunning. We deploy one of the resulting models (namely, the one with the best accuracy) as a webservice. Additionally, we consume the endpoint by making a request to the model by means of the jupyter notebook. 

## Dataset

### Overview
The dataset for this project was obtained from Kaggle. It stablish a level from 1 to 5 indicating a level of alcohol consumption in students during the weekend days. The set of 32 columns includes education information about the students, together with information about their parents ocupation, family relationships among others. Our target colum is well balanced. That is why we use Accuracy for testing. The dataset contains information from 395 students. 

### Task
This is a multilabel classification (from 1 to 5). For this task, we use a Desicion Tree Classifier, where we look for the best parameters by means of Hyperparametertunning tool. On the other hand, we use AutoML for finding the best possible model for this setting.

### Access
In the autoML experiment, we upload the data from local files to the Azure workspace and we call the file from the jutpyter notebook. For the Hyperparameter tunning experiment, we call the data from webfiles, otherwise we would need to call the workspace from the train.py file, and we would need to do manual authentication each time any run is performed.

## Automated ML
For this autoML experiment we set the primary metric to be the Accuracy. On the other hand, the number of folds for cross valitation is set to be 2. The reason for this is that the number of columns in the dataset (<500) is not large enough for each sample to be sufficiently large. Also, we want to save time. Finally, we set the experiment timeout minutes to be 30, for the dataset is not that large and we don't want the experiment to take so long for now.  

### Results
The model obtained from the AutoML experiment was a Voting Ensemble Model. The parameters are the models involved in the ensemble. Namely: XGBoostClassifier, LightGBM, ExtremeRandomTrees, ExtremeRandomTrees, ExtremeRandomTrees, LightGBM, XGBoostClassifier, ExtremeRandomTrees, RandomForest. The weights of this model are given by 0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.18181818181818182 and 0.09090909090909091 respectively. The accuracy obtained in this case was: 0.54.

The model could be improved (acuracy is lower than 60%) by making an improvement of the data celansing, cause in this case the cleansing reduces to convert strings to integers. We could fill the missing values with the mean or by means of other statisical methods, instead of just removing the rows with missing values.

We provide below screenshots of the rundetails and the obtained best model with its parameters and Run ID.

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails.PNG)


![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails2.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model%20ID.PNG)



## Hyperparameter Tuning
We used a Desicion Tree Classifier for this project. The reason is mainly that there are several columns labeles with "yes" and "no", this leads to the possibility of using a decision tree. The rest of the columns are labeled differently, but the values are only integers from 1 to 5 and, many of them actually refer to ordered "levels" of quality of relationships, education among others.

The parameters chosen for tunning in this project are Min_sample_split, i.e. the minimum number of samples required to split an internal node. We chosed to give a range varying among units and tens in order to determine if the model needs several (50) or few samples for splitting.
The second parameter is Max_leaf_nodes, which restricts the number of ending nodes of the tree. Here the model is deciding the expansion capacity of the tree, we want to know if it is better for the tree to expand significantly at the end leaves.



### Results
The best parameters otained from this experiment are:
Max_leaf_nodes=5,
Min_samples_split=10,
and the accuracy achieved for these parameters was 0.42. The model could be improved for sure, since the acuracy is quite low, and I think that we could change to a non-binary tree, cause several columns have labels from 1 to 5. I think a tree with 5 leaves for some of the nodes. However, since some (and not few) of the columns are labeled with 'yes' and 'no' tags, we could analize the data set separately. 

We provide below screenshots of the rundetails and the obtained best model with its parameters:

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/Rundetails1.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/Rundetails2.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/HyperdriveBestModel.PNG)

## Model Deployment
Since the best accuracy for this project was obtained with the AutoML model, this was a Voting Ensemble model which we considered for deployment. Below we show the healthy status after the deployment procedure.



In order to make a request to the endpoint, we provide the URL and Primary Key as follows:

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/Deployment1.PNG)

Then, we provide a sample input:

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/Deployment%202.PNG)

After which we obtain the result:

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/DeploymentRESULT.PNG)


## Screen Recording
https://drive.google.com/file/d/1m_7MIO2_mJ8LTzTL_jfuW8jqCxpH6Q1b/view?usp=sharing

