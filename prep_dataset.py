import os
import shutil
import random
import requests
import zipfile, io

import config

random.seed(config.seed)
num_files_to_move = 150

def load_data():
    base_url = 'https://storage.googleapis.com/kaggle-data-sets/8552/57440/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220131%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220131T190927Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=22bca864a8f8365afd3586b8a442fe999ec251b536eff32b66512c66a9ea9dbcfe4c628a358e29d09c0214425fd9419e322e45816bb61e6e11b60b0ee7eaea5c817e0062331a1e27f8dffce03acb4222a094ee8d6aa377672d3db2cdab12936663eb6678f451742410f74e3fe14af0f666e29dc9857092124a09bec3aacab206050e19ec6fe1cd7e2a0efd75d8157dd37a0d25a8c0bbc30e21ace53650dcff88c346a8611c0b9d116f030afb6b415fc88fac59fc12608b9042e47031aad8cc5d7c0dc1c5a3fa1622188e569f850a5947048f6f1497ece71d25e80b5e1090979029ee9bee190d5ebf9ac0efe2d2e3943caf1a86aa5b7c908eb9fb8dfe39bc687e'
    response = requests.get(base_url)
    response.raw.decode_content = True
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(os.getcwd())


if __name__ == "__main__":
    print('Loading data')
    load_data()    

    print('Data pre-processing')
    hot_dog_files = os.listdir(os.path.join(config.train_dir, config.class_names[0]))

    # move hot_dog class files from train to test
    for file_name in random.sample(hot_dog_files, num_files_to_move):
        shutil.move(os.path.join(config.train_dir, config.class_names[0], file_name),
                    os.path.join(config.test_dir, config.class_names[0]))

    # create validation folder with two class-folders inside
    for class_name in config.class_names:
        os.makedirs(os.path.join(config.val_dir, class_name), exist_ok=True)

    # move 20% of train files to validation
    for class_name in config.class_names:
        source_dir = os.path.join(config.train_dir, class_name)
        source_files = os.listdir(source_dir)
        
        for file_name in random.sample(source_files, len(source_files) // 5):
            shutil.move(os.path.join(source_dir, file_name),
                        os.path.join(config.val_dir, class_name))
    print('Dataset is ready')