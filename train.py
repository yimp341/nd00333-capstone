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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

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
def main():
    x= pd.DataFrame(ds1)
    y= ds1.drop("Walc", inplace=True, axis=1)

    
 #   return x_df, y_df
    
    # Add arguments to script
#x, y = clean_data(ds1)

# TODO: Split data into train and test sets.


    x_train, x_test = train_test_split(x, test_size=0.2)
    y_train, y_test = train_test_split(y, test_size=0.2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("n_estimators:", np.int(args.n-estimators))
    run.log("min_samlples_split", np.int(args.min_samples_split))

    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    model = RandomForestClassifier(n_estimators = args.n_estimators, min_samples_split = args.min_samples_split)
#model = RandomForestClassifier(n_estimators = args.n_estimators)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

