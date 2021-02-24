import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#from sklearn.datasets import load_iris
#from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
ds = Dataset.get_by_name(ws,'alcohol2')
ds1=ds.to_pandas_dataframe()
run = Run.get_context()
#
    # Dict for cleaning data
#    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
#    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
 #   x_df = data.to_pandas_dataframe().dropna()
 #   jobs = pd.get_dummies(x_df.job, prefix="job")
 #   x_df.drop("job", inplace=True, axis=1)
 #   x_df = x_df.join(jobs)
 #   x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
 #   x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
 #   x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
 #   x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
 #   contact = pd.get_dummies(x_df.contact, prefix="contact")
 #   x_df.drop("contact", inplace=True, axis=1)
 #   x_df = x_df.join(contact)
 #   education = pd.get_dummies(x_df.education, prefix="education")
 #   x_df.drop("education", inplace=True, axis=1)
 #   x_df = x_df.join(education)
 #   x_df["month"] = x_df.month.map(months)
 #   x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
 #   x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

def clean_data(data): 
    x=data.to_pandas_dataframe

    schools = {"GP":1, "MS":2}
    sexs = {"F":1, "M":2}
    famsizes = {"LT3":1, "GT3":2}
    Pstatuss = {"A":1, "T":2}
    Mjobs = {"at_home":1, "services":2, "health":3, "teacher":4, "other":5}
    Fjobs = {"at_home":1, "services":2, "health":3, "teacher":4, "other":5}
    reasons = {"course":1, "home":2, "reputation":3, "other":4}



    # Clean and one hot encode data
 #   x_df = data.to_pandas_dataframe().dropna()
 #   jobs = pd.get_dummies(x_df.job, prefix="job")
 #   x_df.drop("job", inplace=True, axis=1)
 #   x_df = x_df.join(jobs)
 #   x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
 #   x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
 #   x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
 #   x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
 #   contact = pd.get_dummies(x_df.contact, prefix="contact")
 #   x_df.drop("contact", inplace=True, axis=1)
 #   x_df = x_df.join(contact)
 #   education = pd.get_dummies(x_df.education, prefix="education")
 #   x_df.drop("education", inplace=True, axis=1)
 #   x_df = x_df.join(education)
    x["school"] = x.school.map(schools)
    x["famsize"] = x.famsize.map(famsizes)
    x["sex"] = x.sex.map(sexs)
    x["Pstatus"] = x.Pstatus.map(Pstatuss)
    x["Mjob"] = x.Mjob.map(Mjobs)
    x["Fjob"] = x.Fjob.map(Fjobs)
    x["reason"] = x.reason.map(reasons)
 #   x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x["schoolsup"] = x.schoolsup.apply(lambda s: 1 if s == "yes" else 0)
    x["famsup"] = x.famsup.apply(lambda s: 1 if s == "yes" else 0)
    x["paid"] = x.paid.apply(lambda s: 1 if s == "yes" else 0)
    x["activities"] = x.activities.apply(lambda s: 1 if s == "yes" else 0)
    x["nursery"] = x.nursery.apply(lambda s: 1 if s == "yes" else 0)
    x["higher"] = x.higher.apply(lambda s: 1 if s == "yes" else 0)
    x["internet"] = x.internet.apply(lambda s: 1 if s == "yes" else 0)
    x["romantic"] = x.romantic.apply(lambda s: 1 if s == "yes" else 0)
return x

def main():

    x = clean_data(ds)
    y= x.drop("Walc", inplace=False, axis=1)

    
 #   return x_df, y_df
    
    # Add arguments to script
#x, y = clean_data(ds1)

# TODO: Split data into train and test sets.


    x_train, x_test = train_test_split(x, test_size=0.2)
    y_train, y_test = train_test_split(y, test_size=0.2)

    parser = argparse.ArgumentParser()
   #max_leaf_nodes
    #parser.add_argument('--n_estimators', type=int, default=100, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_leaf_nodes', type=int, default=10, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("max_leaf_nodes:", np.int(args.max_leaf_nodes))
    run.log("min_samples_split", np.int(args.min_samples_split))

    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    #---this-one--model = RandomForestClassifier(n_estimators = args.n_estimators, min_samples_split = args.min_samples_split)

    model = DecisionTreeClassifier(min_samples_split = args.min_samples_split, max_leaf_nodes = args.max_leaf_nodes).fit(x_train,y_train)
  #model = RandomForestClassifier(n_estimators = args.n_estimators)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

