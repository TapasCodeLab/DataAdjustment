from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# training_file= "C:\\Users\\user\\Desktop\\Project\\training_file.csv"
# result_feature = 'credit_rwa'

def extract_features(training_file, result_feature):
    input_df = pd.read_csv(training_file)
    non_numeric_columns = input_df.select_dtypes(exclude=['number']).columns.tolist()
    d = {}
    for col in non_numeric_columns:
        d[col + '_num'] = col
        input_df[col + '_num'] = pd.factorize(input_df[col])[0]
    numeric_columns = input_df.select_dtypes(include=['number']).columns.tolist()
    corr = input_df.filter(numeric_columns).corr()
    col_list = list(corr.columns)
    res = corr[result_feature]
    feature_list = []
    for col in col_list:
        if res[col] > .1 and res[col] < 1:
            feature_list.append(d.get(col, col))
    f, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, cmap='Blues', annot=True, square=False, ax=ax)
    plt.title('Correlation Matrix for '+result_feature)
    plt.yticks(rotation=45)
    # plt.show()
    return feature_list


