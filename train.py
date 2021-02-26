#pruueba del train.py
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
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#from sklearn.datasets import load_iris
#from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

#ws = Workspace.from_config()
#ds = Dataset.get_by_name(ws,'alcohol')

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '1b944a9b-fdae-4f97-aeb1-b7eea0beac53'
resource_group = 'aml-quickstarts-139488'
workspace_name = 'quick-starts-ws-139488'

workspace = Workspace(subscription_id, resource_group, workspace_name)

ds = Dataset.get_by_name(workspace, name='alcohol')

run = Run.get_context()


def clean_data(dat): 
    xd=dat.to_pandas_dataframe()
    schools = {"GP":1, "MS":2}
    sexs = {"F":1, "M":2}
    addresss = {"U":1, "R":2}
    famsizes = {"LT3":1, "GT3":2}
    Pstatuss = {"A":1, "T":2}
    Mjobs = {"at_home":1, "services":2, "health":3, "teacher":4, "other":5}
    Fjobs = {"at_home":1, "services":2, "health":3, "teacher":4, "other":5}
    guardians = {"mother":1, "father":2, "other":3}
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
    xd["school"] = xd.school.map(schools)
    xd["address"] = xd.address.map(addresss)
    xd["famsize"] = xd.famsize.map(famsizes)
    xd["sex"] = xd.sex.map(sexs)
    xd["Pstatus"] = xd.Pstatus.map(Pstatuss)
    xd["Mjob"] = xd.Mjob.map(Mjobs)
    xd["Fjob"] = xd.Fjob.map(Fjobs)
    xd["reason"] = xd.reason.map(reasons)
    xd["guardian"] = xd.guardian.map(guardians)
 #   x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    xd["schoolsup"] = xd.schoolsup.apply(lambda s: 1 if s == "yes" else 0)
    xd["famsup"] = xd.famsup.apply(lambda s: 1 if s == "yes" else 0)
    xd["paid"] = xd.paid.apply(lambda s: 1 if s == "yes" else 0)
    xd["activities"] = xd.activities.apply(lambda s: 1 if s == "yes" else 0)
    xd["nursery"] = xd.nursery.apply(lambda s: 1 if s == "yes" else 0)
    xd["higher"] = xd.higher.apply(lambda s: 1 if s == "yes" else 0)
    xd["internet"] = xd.internet.apply(lambda s: 1 if s == "yes" else 0)
    xd["romantic"] = xd.romantic.apply(lambda s: 1 if s == "yes" else 0)
    xd=xd.apply(pd.to_numeric, errors='coerce')
    xd=xd.dropna()
    xd=xd.reset_index(drop=True)
    return xd

def main():

    x = clean_data(ds)
    y= x.pop("Walc")


    x_train, x_test = train_test_split(x, test_size=0.2,train_size=0.8,random_state=30)
    y_train, y_test = train_test_split(y, test_size=0.2,train_size=0.8)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_leaf_nodes', type=int, default=10, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    run.log("max_leaf_nodes:", np.int(args.max_leaf_nodes))
    run.log("min_samples_split", np.int(args.min_samples_split))

    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    #---this-one--model = RandomForestClassifier(n_estimators = args.n_estimators, min_samples_split = args.min_samples_split)

    model = DecisionTreeClassifier(min_samples_split = args.min_samples_split, max_leaf_nodes = args.max_leaf_nodes)
    model.fit(x_train,y_train)
#classifier = MultiOutputClassifier(model, n_jobs=-1)
#classifier.fit(x_train,y_train)
  #model = RandomForestClassifier(n_estimators = args.n_estimators)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

