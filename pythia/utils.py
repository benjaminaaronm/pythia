import os


def list_text_files(directory):
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith('.txt'):
            files.append(path)
    return files
