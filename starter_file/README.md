# Student alcohol consuption 

*TODO:* In this project, we aim to predict the alcohol consumption among students, by means of both AutoML and hyperparameter tunning. We deploy one of the resulting models as a webservice. 

## Dataset

### Overview
The dataset for this project was obtained from Kaggle. It stablish a level from 1 to five of alcohol consumption in students. The set of columns includes education information about the students, together with information about their parents ocupation, family relationships among others. 

### Task
This is a multilabel classification (from 1 to 5). For this task we use a desicion tree classifier, where we look for the best parameters by means of Hyperparametertunning tool. On the other hand, we use AutoML for finding the best possible model for this setting.

### Access
In the autoML experiment, we upload the data from local files to the Azure workspace and we call the file from the jutpyter notebook. For the Hyperparameter tunning experiment, we call the data from webfiles, otherwise we would need to call the workspace from the train.py file, and we would need to do manual authentication each time any run is performed.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
For this autoML experiment we set the primary metric to be the Accuracy. On the other hand, the number of folds for cross valitation is set to be 2. The reason for this is that the number of columns in the dataset (<500) is not large enough for each sample to be sufficiently large. Also, we want to save time. Finally, we set the experiment timeout minutes to be 30, for the dataset is not that large and we don't want the experiment to take so long for now.  

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The model obtained from the AutoML experiment was a Voting Ensemble Model. The parameters are the models involved in the ensemble. Namely: XGBoostClassifier, LightGBM, ExtremeRandomTrees, ExtremeRandomTrees, ExtremeRandomTrees, LightGBM, XGBoostClassifier, ExtremeRandomTrees, RandomForest. The weights of this model are given by 0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.18181818181818182 and 0.09090909090909091 respectively. The acuracy obtained in this case was: 0.54.

The model could be improved (acuracy is lower than 60%) by making an improvement of the data celansing, cause in this case the cleansing reduces to convert strings to integers. We could fill the missing values with the mean or by menas of other statisical methods, instead of just dropping the rows with missing values.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails2.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model%20ID.PNG)



## Hyperparameter Tuning
We used a desicion tree classifier for this project. The reason is mainly that there are several columns labeles with "yes" and "no", this leads to a decision tree. The rest of the columns are labeled differently but the values are just integers from 1 to five and many of them actually refer to ordered "levels" like quality of relationships or education. The parameters.

The parameters chosed for tunning in this project are Min_sample_split, i.e. the minimum number of samples required to split an internal node. We chosed to give a range varying among units and tens in order to determine if the model needs several (50) or few samples for splitting.
The second parameter is Max_leaf_nodes, which restricts the number of ending nodes of the tree. Here the model is deciding the expansion capacity of the tree, we want to know if it is better that the tree expands significantly at. the ends



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? 
The best parameters otained from this experiment are:
Max_leaf_nodes=5,
Min_samples_split=10,
and the accuracy achieved for these parameters was 0.42. The model could be improved for sure, since the acuracy is quite low, and I think that we could change to a non-binary tree, cause several columns have labels from 1 to 5. I think a tree with 5 leaves for some of the nodes. However, since some (and not few) of the columns are labeled with 'yes' and 'no' tags, we could analize the data set separately. 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
https://drive.google.com/file/d/1m_7MIO2_mJ8LTzTL_jfuW8jqCxpH6Q1b/view?usp=sharing

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
