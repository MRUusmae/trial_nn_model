import re
import pandas as pd

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace email addresses with 'EMAIL'
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Replace URLs with 'URL'
    text = re.sub(r'http\S+', 'URL', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def load_data(data_path):
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(preprocess_text)
    return df
