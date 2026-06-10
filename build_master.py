import pandas as pd
import os

def build_scratch():
    # 1. Load and Label ISOT
    true_df = pd.read_csv('data/True.csv')
    true_df['label'] = 1 # Human/Real
    fake_df = pd.read_csv('data/Fake.csv')
    fake_df['label'] = 0 # Fake
    isot = pd.concat([true_df, fake_df])[['text', 'label']]

    # 2. Load and Label WELFake 
    # WELFake uses: 1 for Real, 0 for Fake
    welfake = pd.read_csv('data/WELFake_Dataset.csv')[['text', 'label']]

    # 3. Load GenAI Misinfo (is_misinformation: 0=Authentic, 1=Misinfo)
    genai = pd.read_csv('data/generative_ai_misinformation_dataset.csv')
    genai = genai.rename(columns={'is_misinformation': 'label'})
    # Flip labels if necessary to match: 1=Human, 0=AI/Fake
    genai['label'] = genai['label'].map({0: 1, 1: 0})
    genai = genai[['text', 'label']]

    # 4. Merge and Deduplicate
    master = pd.concat([isot, welfake, genai]).dropna()
    master = master.drop_duplicates()
    
    # Save to the expected location
    os.makedirs('data/processed', exist_ok=True)
    master.to_csv('data/processed/master_cleaned.csv', index=False)
    print(f"✅ Created master_cleaned.csv with {len(master)} rows.")

if __name__ == "__main__":
    build_scratch()