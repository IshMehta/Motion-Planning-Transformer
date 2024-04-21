import os

def rename_files_in_directory(directoy):
        for _, dirs, files in os.walk(directoy):
            for file in files:
                if file.endswith('.p'):
                    new_name = ''
                    if file == 'path_20.p':
                        new_name = 'path_0.p'
                    if file == 'path_21.p':
                        new_name = 'path_1.p'
                    if file == 'path_22.p':
                        new_name = 'path_2.p'
                    if file == 'path_23.p':
                        new_name = 'path_3.p'
                    if file == 'path_24.p':
                        new_name = 'path_4.p'
                    
                    old_path = os.path.join(directoy, file)
                    new_path = os.path.join(directoy, new_name)
                    print(old_path, new_path)

                    os.rename(old_path, new_path)
for env in range(500):
    rename_files_in_directory(f"new_maze/train/env{env:06d}")