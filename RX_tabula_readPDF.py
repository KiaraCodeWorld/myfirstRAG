import tabula
import pandas as pd
import json 
#import chromadb
#from langchain.embeddings import HuggingFaceEmbeddings

# Load sentence embeddings (replace with your desired model)
#emb_model = "sentence-transformers/all-MiniLM-L6-v2"
#embeddings = HuggingFaceEmbeddings(model_name=emb_model, cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME'))

# Create a dictionary to map old values to new values
replacement_dict = {
    'PA': 'Prior authorization',
    'QL': 'Quantity Limits',
    'LD': 'Limited distribution'
}

speciality_replace_dict = {
    'SP': 'Speciality Drug'
}

### BlueOptions

pagesMyBlue = list(range(17, 190))
print(pagesMyBlue)

df1  = tabula.read_pdf("BlueOptions.pdf", pages=pagesMyBlue, output_format="dataframe")
BlueOptions_df = pd.concat(df1, ignore_index=True)

#BlueOptions_df["Plan Type"] = "my blue"
BlueOptions_df["Covered for BlueOptions plan"] = "Yes"
BlueOptions_df['Drug_name'] = BlueOptions_df['Unnamed: 0']
BlueOptions_df.drop(columns=['Unnamed: 0'], inplace=True)
BlueOptions_df['Requirements_Limits'] = BlueOptions_df['Unnamed: 3']
BlueOptions_df.drop(columns=['Unnamed: 3'], inplace=True)

BlueOptions_df['Drug_name'] = BlueOptions_df['Drug_name'].replace(r'\n', ' ', regex=True).replace(r'\r', ' ', regex=True)
BlueOptions_df['Requirements_Limits'] = BlueOptions_df['Requirements_Limits'].replace(replacement_dict, regex=True)
BlueOptions_df['Requirements_Limits'] = BlueOptions_df['Requirements_Limits'].fillna("Not Specified")
BlueOptions_df['Specialty'] = BlueOptions_df['Specialty'].fillna("Not Specified").replace(speciality_replace_dict, regex=True)
BlueOptions_df = BlueOptions_df[BlueOptions_df['Drug Tier'] != 'Drug Tier']
BlueOptions_filtered = BlueOptions_df.dropna(subset=['Drug Tier'])

### MY BLUE
pagesMyBlue = list(range(17, 132))
print(pagesMyBlue)

df  = tabula.read_pdf("ValueScriptRxMedGuide.pdf", pages=pagesMyBlue, output_format="dataframe")
myblue_df = pd.concat(df, ignore_index=True)

myblue_df["Plan Type"] = "my blue"
myblue_df["Covered by MyBlue plan"] = "Yes"
myblue_df['Drug_name'] = myblue_df['Unnamed: 0']
myblue_df.drop(columns=['Unnamed: 0'], inplace=True)
myblue_df['Requirements_Limits_MyBlue'] = myblue_df['Unnamed: 3']
myblue_df.drop(columns=['Unnamed: 3'], inplace=True)
myblue_df['Drug Tier for MyBlue'] = myblue_df['Drug Tier']

# Create a dictionary to map old values to new values

myblue_df['Drug_name'] = myblue_df['Drug_name'].replace(r'\n', ' ', regex=True).replace(r'\r', ' ', regex=True)
myblue_df['Requirements_Limits'] = myblue_df['Requirements_Limits'].replace(replacement_dict, regex=True)
myblue_df['Requirements_Limits for MyBlue plan'] = myblue_df['Requirements_Limits'].fillna("Not Specified")
myblue_df['Specialty'] = myblue_df['Specialty'].fillna("Not Specified").replace(speciality_replace_dict, regex=True)
#myblue_df['text'] ='for plan '+ myblue_df['Plan Type'] + 'Drug name: '+ myblue_df['Drug_name'] + 'is allowed '+ myblue_df['Allowed'] + ' and is having Drug tier: ' + myblue_df['Drug Tier'] + ' and Speciality is '+ myblue_df['Specialty'] + ' and requirement/limits are ' + myblue_df['Requirements_Limits'] + '.'
#myblue_df['text'] = myblue_df['text'].replace(r'\n', ' ', regex=True)
myblue_df = myblue_df[myblue_df['Drug Tier'] != 'Drug Tier']
#myblue_df.drop(columns=['text'], inplace=True)
myblue_filtered = myblue_df.dropna(subset=['Drug Tier'])

#myblue_filtered.to_csv('myBlue.csv')

merged_df = pd.merge(BlueOptions_filtered, myblue_filtered[['Drug_name', 'Drug Tier for MyBlue plan','Covered by MyBlue plan']], on='Drug_name', how='left')

print(merged_df.columns)

merged_df.to_csv('Rx.csv')