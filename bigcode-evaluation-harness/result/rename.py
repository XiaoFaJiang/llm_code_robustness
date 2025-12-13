import os
for root, dirs, files in os.walk("."):
    for dir_name in dirs:
        old_path = os.path.join(root, dir_name)
        new_path = os.path.join(root, dir_name.lower())
        os.rename(old_path, new_path)