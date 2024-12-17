import pandas as pd
data = pd.read_csv(r"C:\dev\ml\project\dataframe1.csv")
new_order = ['M', 'G', 'MD', 'W', 'AI', 'S', 'O', 'P','os'] + \
            [col for col in data.columns if col not in ['M', 'G', 'MD', 'W', 'AI', 'S', 'O', 'P','os','Cpu_brand', 'Ram', 'Gpu_brand']] + \
            ['Cpu_brand', 'Ram', 'Gpu_brand']
data_reordered = data[new_order]
output_path = r'C:\dev\ml\project\dataframe2.csv'
data_reordered.to_csv(output_path, index=False)
