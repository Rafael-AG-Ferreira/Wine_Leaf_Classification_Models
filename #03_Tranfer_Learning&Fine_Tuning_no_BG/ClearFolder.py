import os


def clear_folder(folder_path):      # Função para apagar pastas
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)  # Apagar, de forma recursiva, as subpastas
            os.rmdir(file_path)
    pass
