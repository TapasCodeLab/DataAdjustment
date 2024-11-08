import random
import pandas as pd
from datetime import datetime

def calculate(drawn_amt,undrawn_amt,credit_conversion_factor,risk_weight):
    EAD = drawn_amt + (undrawn_amt*credit_conversion_factor)
    Credit_RWA = round(EAD*risk_weight,6)          #∑(Exposure×Risk_Weight)
    return Credit_RWA

def generate_training_file(file_name, number_of_rows):
    cap_id = 100000
    with (open(file_name, 'w+') as file):
        line = "cap_id,category,drawn_amt,undrawn_amt,credit_conversion_factor,risk_weight,credit_rwa\n"
        file.writelines(line)
        for i in range(number_of_rows):
            category = chr(ord('A') + random.randint(0, 25)) + chr(ord('A') + random.randint(0, 25)) + chr(
                ord('A') + random.randint(0, 25))
            drawn_amt= random.randint(10000, 1000000) * 100
            undrawn_amt = random.randint(10000, 1000000) * 100
            credit_conversion_factor= random.random()
            risk_weight = random.choice([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
            credit_rwa = calculate(drawn_amt, undrawn_amt, credit_conversion_factor, risk_weight)
            line = str(cap_id) + ',' + str(category) + ',' + str(drawn_amt) + ',' + str(undrawn_amt) + ',' + str(credit_conversion_factor) + ',' + str(risk_weight) +',' + str(credit_rwa) + "\n"
            file.writelines(line)
            cap_id += 1
        print("Training File created successfully ... ")

def generate_test_input_file(file_name, number_of_rows):
    cap_id = 100000
    with (open(file_name, 'w+') as file):
        line = "cap_id,category,drawn_amt,undrawn_amt,credit_conversion_factor,risk_weight\n"
        file.writelines(line)
        for i in range(number_of_rows):
            category = chr(ord('A') + random.randint(0, 25)) + chr(ord('A') + random.randint(0, 25)) + chr(
                ord('A') + random.randint(0, 25))
            drawn_amt= random.randint(10000, 1000000) * 100
            undrawn_amt = random.randint(10000, 1000000) * 100
            credit_conversion_factor= random.random()
            risk_weight = random.choice([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
            line = str(cap_id) + ',' + str(category) + ',' + str(drawn_amt) + ',' + str(undrawn_amt) + ',' + str(credit_conversion_factor) + ',' + str(risk_weight) + "\n"
            file.writelines(line)
            cap_id += 1
        print("Test input file created successfully ... ")

def generate_test_output_file(input_file, output_file):
    time_now1 = datetime.now()
    df = pd.read_csv(input_file)
    df['credit_rwa'] = calculate(df['drawn_amt'], df['undrawn_amt'], df['credit_conversion_factor'], df['risk_weight'])
    df.to_csv(output_file, index=False)
    print("Test output file created successfully ... ")
    time_now2 = datetime.now()
    print("Time taken to generate test output is: ", (time_now2 - time_now1).total_seconds())

def generate_final_output_file(test_output_file, model1_output_file, model2_output_file, final_output_file):
    time_now1 = datetime.now()
    output_df = pd.read_csv(test_output_file)
    model1_df = pd.read_csv(model1_output_file)
    model2_df = pd.read_csv(model2_output_file)
    temp_df = pd.merge(output_df, model1_df, on=['cap_id','category'])
    temp_df['model1_rwa_diff'] = round(((temp_df['dtm_rwa']-temp_df['credit_rwa'])*100/temp_df['credit_rwa']),4)
    df = pd.merge(temp_df, model2_df, on=['cap_id','category'])
    df['model2_rwa_diff'] = round(((df['rfm_rwa']-df['credit_rwa'])*100/df['credit_rwa']),4)
    df.to_csv(final_output_file, index=False)
    print("Final output file created successfully ... ")
    time_now2 = datetime.now()
    print("Time taken to generate final output is: ", (time_now2 - time_now1).total_seconds())
