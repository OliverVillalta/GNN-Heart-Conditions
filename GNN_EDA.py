import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import Counter
import h5py as h5
from scipy.fft import fft, ifft

#heart_data = pd.read_csv('/')
heart_data = pd.read_csv('/')
#NOTICE ---> This part was made to simply observe the demographics of the dataset. The part after this one is for the modification of the dataset so that all data is within 
# one newly created csv file to prepare for 'PyTorch Geometric. - 01/22/2025

# Data Exloration Stage 
heart_data.loc[heart_data['AHA_Code'] == '1', 'Reading'] = 'Normal'
heart_data.loc[heart_data['AHA_Code'] == '2', 'Reading'] = 'Otherwise Normal' 
heart_data.loc[(heart_data['AHA_Code'] != '1') & (heart_data['ECG_ID'] != '2'), 'Reading'] = 'Abnormal'


abnormal = heart_data[heart_data['Reading'] == 'Abnormal']
ab_count = heart_data['Reading'].value_counts()
ab_sex_count = abnormal['Sex'].value_counts()

diff_colors = {'F': 'red', 'M': 'blue'}

# Data Visualization 
plt.figure(figsize=(10, 6))
plt.bar(ab_sex_count.index, ab_sex_count.values, color=[diff_colors[sex] for sex in ab_sex_count.index])
plt.xlabel('Sex')
plt.ylabel('Count of Abnormal Readings')
plt.title('Abnormal Heart Readings Differentiated By Sex')
plt.show()

print(heart_data['Age'].describe())

plt.figure(figsize=(20, 6))
plt.hist(abnormal['Age'], edgecolor = 'black', bins = 85, color='blue')
plt.xticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
plt.xlabel('Age')
plt.ylabel('Count of Abnormal Readings')
plt.title('Abnormal Heart Readings Differentiated by Age')
plt.show()

sex_counts = heart_data['Sex'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(sex_counts.index, sex_counts.values, color=[diff_colors[sex] for sex in sex_counts.index])
plt.xlabel('Sex')
plt.ylabel('Number of Patients')
plt.title('Sex Distribution of Patients')
plt.show()

plt.figure(figsize=(20, 12))
plt.hist(heart_data['Age'], edgecolor = 'black', bins = 85, color = 'blue')
plt.xticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Age Distribution of Patients')
plt.show()

plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.2f%%', colors=[diff_colors[sex] for sex in sex_counts.index])
plt.show()

plt.pie(ab_count.values, labels=ab_count.index, autopct='%0.4f%%', colors=['green', 'red'])
plt.show()

# Feature Engineering Stage

heart_data = heart_data.dropna(subset=['Reading'])

# Convert strings that look like lists into actual Python lists
heart_data['Reading'] = heart_data['Reading'].apply(ast.literal_eval)

# Flatten the list of all diagnosed conditions
all_conditions = [condition for sublist in heart_data['Reading'] for condition in sublist]
condition_counts = Counter(all_conditions)
count_df = pd.DataFrame(condition_counts.items(), columns=['Condition', 'Count'])
count_df = count_df.sort_values(by='Count', ascending=False)
plt.figure(figsize=(12, 7))
plt.bar(count_df['Condition'], count_df['Count'], color='teal')
plt.title('Frequency of Diagnosed Conditions')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

aha_coding = pd.read_csv('C:\\Users\\School Profile\\Documents\\Senior Project Thesis\\records\\code.csv')

def remove_modifier(series):

    primary_codes = []
    copy = ""
    result = []
    print('Processing: ' + series)
    for stuff in series:
        print('Processing')
        if stuff == ';' or stuff == '+':
            primary_codes.append(int(copy))
            primary_codes.append(stuff)
            copy = ""
        else:
            copy+=stuff
    if copy != ' ':
            primary_codes.append(int(copy))
    while primary_codes:
        if primary_codes[-1] == ';' or primary_codes[-1] == '+':
            primary_codes.pop()
        elif primary_codes[-1] in result:
            primary_codes.pop()
        else:
            result.append(int(primary_codes.pop()))
    return result


def add_row(r):
    primary_code_descriptions = []
    for code in primary_codes[r.name]:
        matching_row = aha_coding[aha_coding.Code == code]
        if not matching_row.empty:
            description = matching_row.Description.iloc[0]
            primary_code_descriptions.append(description)
    return primary_code_descriptions

# Storing updated metadata to another csv file to reduce need to do this every single time I want to run the model. 
heart_data['AHA_Code'] = heart_data['AHA_Code'].str.replace(r'\s+', '', regex=True) 
heart_data['AHA_Code'] = heart_data['AHA_Code'].astype(str)
codes = heart_data.AHA_Code.str.split('[;+]', expand=True)
codes.columns = ['Code_1', 'Code_2', 'Code_3', 'Code_4', 'Code_5', 'Code_6', 'Code_7', 'Code_8', 'Code_9']
print(codes.loc[codes['Code_1'] == ''])
primary_codes = heart_data['AHA_Code'].map(remove_modifier)
prepared_data = pd.concat([heart_data, codes], axis = 1)
prepared_data['Reading'] = prepared_data.apply(add_row, axis=1)
prepared_data = prepared_data.drop(columns=['AHA_Code'])
print(prepared_data.head(10))

prepared_data.to_csv('/', index=False) 


final_data = pd.read_csv('/')
final_data['Code_1'] = final_data['Code_1'].astype(int)
final_data['Code_2'] = final_data['Code_1'].astype(int)
final_data['Code_3'] = final_data['Code_1'].astype(int)
final_data['Code_4'] = final_data['Code_1'].astype(int)
final_data['Code_5'] = final_data['Code_1'].astype(int)
final_data['Code_6'] = final_data['Code_1'].astype(int)
final_data['Code_7'] = final_data['Code_1'].astype(int)
final_data['Code_8'] = final_data['Code_1'].astype(int)
final_data['Code_9'] = final_data['Code_1'].astype(int)

# ECG & FFT Graph Generation (Visual)
def denoise_signal(ecg_signal, fs, low_cutoff, high_cutoff):
    N = len(ecg_signal)
    fft_signal = fft(ecg_signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_signal[np.abs(freqs) < low_cutoff] = 0
    fft_signal[np.abs(freqs) > high_cutoff] = 0
    denoised_signal = np.real(ifft(fft_signal))
    return denoised_signal


fs = 500 
low_cutoff = 0.5
high_cutoff = 40
with h5.File(f'/', 'r') as f:
    signal = f['ecg'][()]
    for i in range(12):
        #denoised = denoise_signal(signal[i], fs, low_cutoff, high_cutoff)
        plt.plot(signal[i])
    plt.title('Raw ECG Signal')
    plt.tight_layout()
    plt.show()

