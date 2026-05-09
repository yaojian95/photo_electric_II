import csv
import json
import os

csv_path = r'E:\multi_source_info\data_dir\20260325_yinshan\2026_矿石数据采集-0325_银山铜矿_114.csv'
json_path = r'E:\multi_source_info\data_dir\disk_grades.json'

new_grades = {}

# Try reading with utf-8-sig first, fallback to gbk
try:
    f = open(csv_path, 'r', encoding='utf-8-sig')
    reader = csv.reader(f)
    next(reader)
except UnicodeDecodeError:
    f.close()
    f = open(csv_path, 'r', encoding='gbk')
    reader = csv.reader(f)

# Skip headers (first 3 rows)
f.seek(0)
for _ in range(3):
    next(reader)

for row in reader:
    if len(row) < 13: 
        continue
    disk_id_str = row[0].strip()
    if not disk_id_str.isdigit(): 
        continue
        
    cu_str = row[10].strip()
    fe_str = row[11].strip()
    s_str = row[12].strip()
    
    if cu_str and fe_str and s_str:
        try:
            new_grades[disk_id_str] = [float(cu_str), float(fe_str), float(s_str)]
        except ValueError:
            pass

f.close()

if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
else:
    full_config = {}

full_config["20260325"] = new_grades

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(full_config, f, indent=4)

print(f"Successfully added {len(new_grades)} records to disk_grades.json for date 20260325.")
