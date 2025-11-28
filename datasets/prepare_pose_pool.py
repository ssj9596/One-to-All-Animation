import pandas as pd
import os

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opensource_pose_pool')
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"📂 工作目录: {os.getcwd()}")

CSV_A = 'images_pose_pool_18_16pt.csv'
CSV_B = 'images_pose_pool_24_18pt.csv'


print(f"\n📊 读取 CSV 文件...")
df_a = pd.read_csv(CSV_A)
df_b = pd.read_csv(CSV_B)


print(f"\n🔄 转换为绝对路径...")
for col in ['pose_path']:
    df_a[col] = df_a[col].apply(lambda x: os.path.abspath(x) if isinstance(x, str) else x)
    df_b[col] = df_b[col].apply(lambda x: os.path.abspath(x) if isinstance(x, str) else x)


print(f"\n✅ 检查文件是否存在...")
for col in ['pose_path']:
    df_a[col + '_exists'] = df_a[col].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)
    print(f"   {CSV_A} 存在文件数量: {df_a[col + '_exists'].sum()} / {len(df_a)}")
    df_b[col + '_exists'] = df_b[col].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)
    print(f"   {CSV_B} 存在文件数量: {df_b[col + '_exists'].sum()} / {len(df_b)}")

df_a.drop(columns=[c for c in df_a.columns if c.endswith('_exists')], inplace=True)
df_b.drop(columns=[c for c in df_b.columns if c.endswith('_exists')], inplace=True)

output_a = CSV_A.replace(".csv", "_abs.csv")
output_b = CSV_B.replace(".csv", "_abs.csv")

df_a.to_csv(output_a, index=False)
df_b.to_csv(output_b, index=False)

print(f"   {os.path.abspath(output_a)}")
print(f"   {os.path.abspath(output_b)}")
