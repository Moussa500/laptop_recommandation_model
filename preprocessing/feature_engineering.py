import pandas as pd
df = pd.read_csv(r"C:\dev\ml\project\dataframe.csv")
def determine_usages(row):
    gaming = 0
    office = 0
    school = 0
    ai = 0
    mobile_dev = 0
    web_dev = 0
    multitasking = 0
    cpu_brand = str(row['Cpu_brand']).lower()
    gpu_brand = str(row['Gpu_brand']).lower()
    ram = row['Ram']  
    if ('i7' in cpu_brand or 'i9' in cpu_brand) and ('nvidia' in gpu_brand or 'amd' in gpu_brand) and ram >= 16:
        gaming = 1
        ai=1
    if ('nvidia' in gpu_brand or 'amd' in gpu_brand) and ram >= 16:
        mobile_dev = 1
    if ram >= 32 or 'i7' in cpu_brand or 'i9' in cpu_brand:
        ai = 1
        multitasking = 1
    if ram >= 16:
        web_dev = 1
    if 'intel' in gpu_brand or 'Dual Core' in cpu_brand or 'i3' in cpu_brand: 
        school = 1
        office = 1
        multitasking = 1  
    return pd.Series([multitasking, gaming, mobile_dev, web_dev, ai, school, office])

df[['M', 'G', 'MD', 'W', 'AI', 'S', 'O']] = df.apply(determine_usages, axis=1)
output_path = r'C:\dev\ml\project\dataframe1.csv'
df.to_csv(output_path, index=False)

output_path
