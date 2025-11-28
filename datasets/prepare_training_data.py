import pandas as pd
import os
from pandarallel import pandarallel
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

pandarallel.initialize(progress_bar=True)

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opensource_dataset')
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"📂 工作目录: {os.getcwd()}")

def unzip_one(zip_path, extract_dir):
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        common_prefix = os.path.commonprefix(names).split('/')[0] if names else ''
        if common_prefix.lower() == zip_name.lower():
            target_dir = extract_dir
        else:
            target_dir = os.path.join(extract_dir, zip_name)
            os.makedirs(target_dir, exist_ok=True)

        zf.extractall(target_dir)
    return f"✅ {zip_name}.zip 已解压到 {target_dir}"

def unzip_all(base_dir=".", extract_dir=None, max_workers=8):
    if extract_dir is None:
        extract_dir = base_dir

    zip_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".zip")
    ]

    if not zip_files:
        print("⚠️  未找到 zip 文件")
        return

    print(f"🔍 共发现 {len(zip_files)} 个 zip 文件，开始多线程解压…")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(unzip_one, z, extract_dir): z for z in zip_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
                results.append(result)
            except Exception as e:
                print(f"❌ 解压失败: {futures[future]} - {e}")

    print("🎉 所有文件已解压完成！")

# unzip
unzip_all(base_dir=".", max_workers=8)

VID_CSV = 'opensource_data_vid.csv'
IMG_CSV = 'opensource_data_fashion_img.csv'
OUTPUT_CSV = 'combined_imgvid_dataset.csv'

# vid:img = 6:1
GROUP_RATIO = 6

# -----------------------------
# 读取数据
# -----------------------------
print(f"\n📊 读取数据...")
df_vid = pd.read_csv(VID_CSV)
df_img = pd.read_csv(IMG_CSV)

# -----------------------------
# 转为绝对路径
# -----------------------------
print(f"🔄 转换为绝对路径...")
for col in ['path', 'pose_path']:
    df_vid[col] = df_vid[col].apply(lambda x: os.path.abspath(x) if isinstance(x, str) else x)

for col in ['path', 'target_path', 'pose_path']:
    df_img[col] = df_img[col].apply(lambda x: os.path.abspath(x) if isinstance(x, str) else x)

# 检查文件是否存在
print(f"\n✅ 检查视频文件是否存在...")
for col in ['path', 'pose_path']:
    df_vid[col + '_exists'] = df_vid[col].parallel_apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)
    print(f"   {col} 存在文件数量: {df_vid[col + '_exists'].sum()} / {len(df_vid)}")

print(f"\n✅ 检查图片文件是否存在...")
for col in ['path', 'target_path', 'pose_path']:
    df_img[col + '_exists'] = df_img[col].parallel_apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)
    print(f"   {col} 存在文件数量: {df_img[col + '_exists'].sum()} / {len(df_img)}")

df_vid.drop(columns=[c for c in df_vid.columns if c.endswith('_exists')], inplace=True)
df_img.drop(columns=[c for c in df_img.columns if c.endswith('_exists')], inplace=True)


n_vid = len(df_vid)
n_img = len(df_img)

GROUP_NUMBER = int(n_vid / GROUP_RATIO) 
GROUP_SIZE = max(1, n_img // GROUP_NUMBER)

print(f"\n📈 视频数量: {n_vid}, 图片数量: {n_img}")
print(f"📈 计算得到 GROUP_SIZE: {GROUP_SIZE}")

# -----------------------------
# 对 img 数据分组
# -----------------------------
print(f"\n🔄 对图片数据进行分组...")
fields = ['height', 'width', 'aspect_ratio', 'resolution']
df_img_sorted = df_img.sort_values(fields + ['path']).reset_index(drop=True)

group_rows       = []
bad_rows         = []
group_id_counter = 0

for combo, subdf in df_img_sorted.groupby(fields, sort=False):
    n           = len(subdf)
    n_groups    = n // GROUP_SIZE
    remainder   = n % GROUP_SIZE
    idxs        = subdf.index.to_list()

    for g in range(n_groups):
        slice_idx = idxs[g*GROUP_SIZE:(g+1)*GROUP_SIZE]
        group     = subdf.loc[slice_idx]

        first = group.iloc[0]

        img_pairs = [
            {
                'src_path'     : row['path'],
                'tgt_path'     : row['target_path'],
                'tgt_pose_path': row['pose_path'],
                'qwen_caption' : row['qwen_caption'] if 'qwen_caption' in row and pd.notnull(row['qwen_caption']) else "",
                'text'         : row['text'] if 'text' in row and pd.notnull(row['text']) else ""
            }
            for _, row in group.iterrows()
        ]

        group_rows.append(
            {
                'path'         : f'image_{group_id_counter+2800:05d}',
                'img_pairs'    : img_pairs,
                'height'       : int(first['height']),
                'width'        : int(first['width']),
                'aspect_ratio' : first['aspect_ratio'],
                'resolution'   : first['resolution'],
                'num_frames'   : 1
            }
        )
        group_id_counter += 1

    if remainder:
        bad_rows.extend(subdf.loc[idxs[-remainder:]].to_dict('records'))

df_groups = pd.DataFrame(group_rows)
df_bad    = pd.DataFrame(bad_rows)

print(f"📊 生成 df_groups: {len(df_groups)} 条, df_bad: {len(df_bad)} 条")

# -----------------------------
# 合并数据
# -----------------------------
df_final = pd.concat([df_groups, df_vid], ignore_index=True, sort=False)
print(f"📊 最终合并后的数据集: {len(df_final)} 条")

# -----------------------------
# 保存 CSV
# -----------------------------
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 已保存为 {OUTPUT_CSV}")
print(f"📂 完整路径: {os.path.abspath(OUTPUT_CSV)}")