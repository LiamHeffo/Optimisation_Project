import os
import shutil

def delete_directory():

    folder_name_list = []
    for i in range(110000):
        folder_name_list.append(f"DEAP_tests_{i}")

    current_directory = os.getcwd()

    for item in os.listdir(current_directory):
        print(item)
        if os.path.isdir(os.path.join(current_directory, item)) and item in folder_name_list:

            shutil.rmtree(os.path.join(current_directory, item))
            print(f"Deleted folder: {item}")


delete_directory()