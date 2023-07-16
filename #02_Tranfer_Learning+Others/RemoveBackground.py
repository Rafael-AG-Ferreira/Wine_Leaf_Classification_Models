import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

done = False

# Caminho da pasta raíz
root_folder = '../../DataV3'

# Nome da pasta de output
output_root_folder = root_folder + '_no_bg'

# Criar a pasta output
os.makedirs(output_root_folder, exist_ok=True)

# Iterar pela pasta raíz e pelas suas subpastas
for dirpath, dirnames, filenames in os.walk(root_folder):
    # Obtenção do caminho relativo
    relative_dirpath = os.path.relpath(dirpath, root_folder)

    if not done:
        # Impressão do nome da pasta que está a ser trabalhada
        folder_name = os.path.basename(dirpath)
        print(f"Pasta {folder_name}:")
        done = True

    # Iteração pelas diferentes pastas (classes) na pasta corrente
    for class_name in dirnames:
        # Obtenção do caminho da pasta
        class_dirpath = os.path.join(dirpath, class_name)

        # Obtenção dos ficheiros na pasta
        class_filenames = os.listdir(class_dirpath)

        # Criação da correspondente pasta de output
        output_class_folder = os.path.join(output_root_folder, relative_dirpath, class_name)
        os.makedirs(output_class_folder, exist_ok=True)

        # Criação de uma barra de progresso para cada classe da pasta
        with tqdm(total=len(class_filenames), desc=f"\t\t{class_name}", unit='file') as pbar:
            for file_index, filename in enumerate(class_filenames, start=1):
                # Obtenção do nome e extenção do ficheiro original
                file_name, ext = os.path.splitext(filename)
                original_file_type = ext[1:]

                if original_file_type.lower() in ['jpg', 'jpeg', 'png', 'gif']:
                    # Obter o caminho da imagem (input)
                    input_path = os.path.join(class_dirpath, filename)

                    # Criação do nome e caminho da imagem de output com o nome original + "no_bg"
                    output_filename = file_name + '_' + original_file_type + '_no_bg.png'
                    output_path = os.path.join(output_class_folder, output_filename)

                    # Carregar a imagem de input e remover o fundo
                    input_image = Image.open(input_path)
                    output_image = remove(input_image)

                    # Guardar a imagem de output
                    output_image.save(output_path)

                # Dá update da progress bar na classe corrente
                pbar.update(1)
