import pandas as pd
import numpy as np

#Load the dataset from an Excel file
#Make sure to provide the correct path to your Excel file
file = 'file_here or path'  # Change this to your actual file path
cleaned_df = pd.read_excel(file)

#Check the structure of the data
print(cleaned_df.head())

#Function to create bootstrap samples
def create_bootstrap_samples(cleaned_df, n_samples=100, sample_size=1000):
    bootstrap_samples = []
    for _ in range(n_samples):
        # Create a bootstrap sample by sampling with replacement
        bootstrap_sample = cleaned_df.sample(n=sample_size, replace=True, random_state=np.random.randint(0, 10000))
        bootstrap_samples.append(bootstrap_sample)
    return bootstrap_samples

#Generate bootstrap samples
n_bootstrap_samples = 10  # Number of bootstrap samples to create
sample_size = 3000  # Size of each bootstrap sample (adjust as needed)
bootstrap_samples = create_bootstrap_samples(cleaned_df, n_samples=n_bootstrap_samples, sample_size=sample_size)

#Example: Displaying the first bootstrap sample
print(bootstrap_samples[0])
