import tabula
import pandas as pd

#import chromadb
#from langchain.embeddings import HuggingFaceEmbeddings

# Load sentence embeddings (replace with your desired model)
#emb_model = "sentence-transformers/all-MiniLM-L6-v2"
#embeddings = HuggingFaceEmbeddings(model_name=emb_model, cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME'))

pages = list(range(17, 132))

print(pages)


#df  = tabula.read_pdf("table.pdf", pages="all", output_format="dataframe", stream=True)
#df  = tabula.read_pdf("ValueScriptRxMedGuide_split.pdf", pages="all", output_format="dataframe")
df  = tabula.read_pdf("ValueScriptRxMedGuide.pdf", pages=pages, output_format="dataframe")


combined_df = pd.concat(df, ignore_index=True)

combined_df["Plan Type"] = "my blue"
combined_df["Allowed"] = "Yes"
combined_df['Drug_name'] = combined_df['Unnamed: 0']
combined_df.drop(columns=['Unnamed: 0'], inplace=True)
combined_df['Requirements_Limits'] = combined_df['Unnamed: 3']
combined_df.drop(columns=['Unnamed: 3'], inplace=True)
# Create a dictionary to map old values to new values
replacement_dict = {
    'PA': 'Prior authorization',
    'QL': 'Quantity Limits',
    'LD': 'Limited distribution'
}

speciality_replace_dict = {
    'SP': 'Speciality Drug'
}
combined_df['Drug_name'] = combined_df['Drug_name'].replace(r'\n', ' ', regex=True)
combined_df['Requirements_Limits'] = combined_df['Requirements_Limits'].replace(replacement_dict, regex=True)
combined_df['Requirements_Limits'] = combined_df['Requirements_Limits'].fillna("Not Specified")
combined_df['Specialty'] = combined_df['Specialty'].fillna("Not Specified").replace(speciality_replace_dict, regex=True)
combined_df['text'] ='for plan '+ combined_df['Plan Type'] + 'Drug name: '+ combined_df['Drug_name'] + 'is allowed '+ combined_df['Allowed'] + ' and is having Drug tier: ' + combined_df['Drug Tier'] + ' and Speciality is '+ combined_df['Specialty'] + ' and requirement/limits are ' + combined_df['Requirements_Limits'] + '.'
combined_df['text'] = combined_df['text'].replace(r'\n', ' ', regex=True)
combined_df = combined_df[combined_df['Drug Tier'] != 'Drug Tier']

combined_df.to_csv("myBlue.csv", index=False)

print("Number of pages in DataFrame:", len(combined_df))




