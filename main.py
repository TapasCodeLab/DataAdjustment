# from train_models import train_model
from feature_extract import extract_features
from file_creation import generate_test_input_file, generate_test_output_file, generate_final_output_file, \
    generate_training_file
from models import decision_tree_model_predict, random_forest_model_predict, train_model
from datetime import datetime
from plot_result import plot_data

training_file = "C:\\Users\\user\\Desktop\\Project\\training_file.csv"
input_file = "C:\\Users\\user\\Desktop\\Project\\input_file.csv"
output_file = "C:\\Users\\user\\Desktop\\Project\\output_file.csv"
dtm_output_file = "C:\\Users\\user\\Desktop\\Project\\dtm_output_file.csv"
rfm_output_file = "C:\\Users\\user\\Desktop\\Project\\rfm_output_file.csv"
final_output_file = "C:\\Users\\user\\Desktop\\Project\\final_output_file.csv"

# INPUTS TO THE PROCESS
training_size = 10000
input_size = 10000
result_feature = 'credit_rwa'
key_features = ['cap_id', 'category']

# Process
print("Generating training file...")
generate_training_file(training_file, training_size)
input_feature = extract_features(training_file, result_feature)

train_model(training_file, input_feature, result_feature)
generate_test_input_file(input_file, input_size)
generate_test_output_file(input_file, output_file)

print("Prediction Started ...")
decision_tree_model_predict(input_file, dtm_output_file, input_feature, key_features)
random_forest_model_predict(input_file, rfm_output_file, input_feature, key_features)
generate_final_output_file(output_file, dtm_output_file, rfm_output_file, final_output_file)
plot_data(final_output_file, input_size // 100, 'cap_id', 'credit_rwa', 'dtm_rwa', 'RWA by CAP_ID')
plot_data(final_output_file, input_size // 100, 'cap_id', 'credit_rwa', 'rfm_rwa', 'RWA by CAP_ID')
