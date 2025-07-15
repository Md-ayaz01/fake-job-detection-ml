import pandas as pd
import os

class DataIngestion:
    def __init__(self, main_data_path, extra_data_path, output_path):
        self.main_data_path = main_data_path
        self.extra_data_path = extra_data_path
        self.output_path = output_path

    def combine_data(self):
        df_main = pd.read_csv(self.main_data_path)
        df_extra = pd.read_csv(self.extra_data_path)

        # Combine title + description automatically
        def merge_text(df):
            text_cols = [col for col in df.columns if 'title' in col.lower() or 'description' in col.lower()]
            df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
            return df[['text', 'fraudulent']]
        
        df_main = merge_text(df_main)
        df_extra = merge_text(df_extra)

        df_combined = pd.concat([df_main, df_extra], ignore_index=True)

        # Remove duplicate headers if accidentally included
        df_combined = df_combined[df_combined['fraudulent'] != 'fraudulent']

        os.makedirs(self.output_path, exist_ok=True)
        df_combined.to_csv(os.path.join(self.output_path, 'combined_data.csv'), index=False)

        print("✅ Data Ingestion Completed. Saved combined_data.csv")
        return df_combined
