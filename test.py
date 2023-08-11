import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Load your dataset
data = pd.read_csv('testdata.csv')

# Create a SingleTableMetadata object and detect metadata from your dataframe
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Create an instance of the CopulaGANSynthesizer model
synthesizer = CopulaGANSynthesizer(metadata)

# Fit the model to your data
synthesizer.fit(data)

# Generate synthetic data
synthetic_data = synthesizer.sample(100000)

# Save the synthetic data to a CSV file
synthetic_data.to_csv('synthetic_data.csv', index=False)