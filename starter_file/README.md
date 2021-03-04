*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* In this project, we aim to predict the alcohol consumption among students, by means of both AutoML and hyperparameter tunning. We deploy one of the resulting models as a webservice. 

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: The dataset for this project was obtained from Kaggle. It stablish a level from 1 to five of alcohol consumption in students. The set of columns includes education information about the students, together with information about their parents ocupation, family relationships among others. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
This is a multilabel classification (from 1 to 5). For this task we use a desicion tree classifier, where we look for the best parameters by means of Hyperparametertunning tool. On the other hand, we use AutoML for finding the best possible model for this setting.

### Access
*TODO*: Explain how you are accessing the data in your workspace.
In the autoML experiment, we upload the data from local files to the Azure workspace and we call the file from the jutpyter notebook. For the Hyperparameter tunning experiment, we call the data from webfiles, otherwise we would need to call the workspace from the train.py file, and we would need to do manual authentication each time any run is performed.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
For this autoML experiment we set the primary metric to be the Accuracy. On the other hand, the number of folds for cross valitation is set to be 2. The reason for this is that the number of columns in the dataset (<500) is not large enough for each sample to be sufficiently large. Also, we want to save time. Finally, we set the experiment timeout minutes to be 30, for the dataset is not that large and we don't want the experiment to take so long for now.  

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The model obtained from the AutoML experiment was a Voting Ensemble Model. The parameters are the models involved in the ensemble. Namely: XGBoostClassifier, LightGBM, ExtremeRandomTrees, ExtremeRandomTrees, ExtremeRandomTrees, LightGBM, XGBoostClassifier, ExtremeRandomTrees, RandomForest. The weights of this model are given by 0.09090909090909091, 0.09090909090909091, 0.18181818181818182, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.18181818181818182 and 0.09090909090909091 respectively. The acuracy obtained in this case was: 0.51.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20rundetails2.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model.PNG)

![alt text](https://github.com/yimp341/nd00333-capstone/blob/master/AutoML%20Besto%20Model%20ID.PNG)



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
