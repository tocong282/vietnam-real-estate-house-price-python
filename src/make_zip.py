import zipfile, glob, os
os.chdir('deliverables_task1')
with zipfile.ZipFile('task1_files.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ['data_dictionary.xlsx', 'report_task1.docx', 'report_task1.md'] + sorted(glob.glob('0*.csv')):
        z.write(f)
print(zipfile.ZipFile('task1_files.zip').namelist())
print(round(os.path.getsize('task1_files.zip') / 1024, 1), 'KB')
