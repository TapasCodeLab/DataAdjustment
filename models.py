import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(random_state=42)

def decision_tree_model_train(training_file, input_features, result_feature):
    time_now1 = datetime.now()
    df_train = pd.read_csv(training_file)
    X_train = df_train[input_features]
    y_train = df_train[result_feature]
    decision_tree_model.fit(X_train, y_train)
    joblib.dump(decision_tree_model, 'decision_tree_model.joblib')
    time_now2 = datetime.now()
    print("Time taken to train the decision tree model is: ", (time_now2 - time_now1).total_seconds())


def random_forest_model_train(training_file,input_features,result_feature):
    time_now1 = datetime.now()
    df_train = pd.read_csv(training_file)
    X_train = df_train[input_features]
    y_train = df_train[result_feature]
    random_forest_model.fit(X_train, y_train)
    joblib.dump(random_forest_model, 'random_forest_model.joblib')
    time_now2 = datetime.now()
    print("Time taken to train the random forest model is: ", (time_now2 - time_now1).total_seconds())


def decision_tree_model_predict(input_file, output_file, input_features, key_features):
    time_now1 = datetime.now()
    df_test = pd.read_csv(input_file)
    new_df = pd.DataFrame(df_test)
    decision_tree_model = joblib.load('decision_tree_model.joblib')
    time_now2 = datetime.now()
    new_df['dtm_rwa'] = decision_tree_model.predict(new_df[input_features])
    kf = key_features[:]
    kf.append('dtm_rwa')
    new_df[kf].to_csv(output_file, index=False)
    time_now3 = datetime.now()
    print("Time taken to Load and predict using the decision tree model is: ", (time_now3 - time_now1).total_seconds())
    print("Time taken to Only predict using the decision tree model is: ", (time_now3 - time_now2).total_seconds())


def random_forest_model_predict(input_file, output_file, input_features, key_features):
    time_now1 = datetime.now()
    df_test = pd.read_csv(input_file)
    new_df = pd.DataFrame(df_test)
    random_forest_model = joblib.load('random_forest_model.joblib')
    time_now2 = datetime.now()
    new_df['rfm_rwa'] = random_forest_model.predict(new_df[input_features])
    kf = key_features[:]
    kf.append('rfm_rwa')
    new_df[kf].to_csv(output_file, index=False)
    time_now3 = datetime.now()
    print("Time taken to Load and predict using the random forest model is: ", (time_now3 - time_now1).total_seconds())
    print("Time taken to Only predict using the random forest model is: ", (time_now3 - time_now2).total_seconds())

def train_model(training_file, input_features, result_feature):
    print("Training DT model...")
    decision_tree_model_train(training_file, input_features, result_feature)
    print("Training RF model...")
    random_forest_model_train(training_file, input_features, result_feature)