import os
import cv2
import pickle
import insightface
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial.distance import cosine


def extract_embeddings(data_folder, save_path="data/embed/embeddings.pkl", nb_imgs: int=2000):
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0, det_size=(224, 224))
    
    # Vérifier si le fichier d'embeddings existe déjà
    if os.path.exists(save_path):
        try:
            with open(save_path, "rb") as f:
                embed = pickle.load(f)
            print("Embeddings chargés depuis le fichier.")
            return embed
        except (EOFError, pickle.UnpicklingError):
            print("Fichier corrompu. Recalcul des embeddings")
    
    def get_face_embedding(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur : Impossible de charger {image_path}")
            return None
        
        faces = model.get(img)
        if len(faces) > 0:
            return faces[0].embedding
        return None
    
    embed = {}
    image_list = os.listdir(data_folder)[:nb_imgs]
    
    for image in tqdm(image_list, desc="Extraction des embeddings", unit="image"):
        image_path = os.path.join(data_folder, image)
        embed[image] = get_face_embedding(image_path)
        
    with open(save_path, "wb") as f:
        pickle.dump(embed, f)
    
    return embed

def get_pairs(embed: dict[str, list], upper_bound: float, lower_bound: float, nb_pairs: int):

    unused_images = embed.copy()
    
    pairs = []

    for _ in tqdm(range(nb_pairs)):

        paired = False
        while not paired and len(unused_images) > 0:
        
            src: str = next(iter(unused_images))
            src_embedding = np.array(embed[src]).flatten()
            del unused_images[src]

            for target, value in unused_images.copy().items():

                if value is None:
                    del unused_images[target]
                    continue
                
                np_value = np.array(value).flatten()
                similarity = 1 - cosine(src_embedding, np_value)
                
                if similarity > 0.75: #target is probably the same person as src
                    del unused_images[target]

                elif similarity < upper_bound and similarity > lower_bound:

                    pairs.append({
                        'source': src.replace(".jpg", "").strip(),
                        'target': target.replace(".jpg", "").strip(),
                        'similarity': round(similarity, 3)
                    })
                    
                    paired = not paired

                    del unused_images[target]
                    break
        
            if not paired:
                print(f"Image {src} not paired. Continuing with the next one.")

    return pairs

def get_reals(embed: dict[str, list[float]], pairs: list[dict], nb_imgs: int):
    all_ids = [id.replace(".jpg", "") for id in embed.keys()]
    
    id_target = [pair['target'] for pair in pairs]
    id_source = [pair['source'] for pair in pairs]
    used_ids = id_target + id_source

    real_ids = list(set(all_ids) - set(used_ids))[: nb_imgs]

    return real_ids

def create_csv(pairs: list[dict], real_ids: list[float]) -> pd.DataFrame:
    frac_pairs = len(pairs)//3

    df = pd.DataFrame(columns=["model", "source", "target", "similarity"])

    for i, model in enumerate(["Thomas", "Julian", "Florian"]):

        df_model = pd.DataFrame(
            pairs[i*frac_pairs:(i+1)*frac_pairs]
        )
        df_model['model'] = model

        df = pd.concat([df, df_model], ignore_index=True)

    real_dict = [{
        'model': 'Real',
        'source': id_,
        'target': id_,
        'similarity': 0
    } for id_ in real_ids]

    df_real = pd.DataFrame(real_dict)

    df = pd.concat([df, df_real], ignore_index=True)

    return df

def split_celeba(
        path_csv: str,
        path_embed: str,
        upper_bound: float = 0.20,
        lower_bound: float = 0.25,
        nb_pairs: int = 6000,
        add_reals: bool = True
    ):

    with open(path_embed, 'rb') as file:
        embed = pickle.load(file)

    pairs = get_pairs(
        embed,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        nb_pairs=nb_pairs
    )
    
    # Checking unicity of images in the dataset
    id_target = [pair['target'] for pair in pairs]
    id_source = [pair['source'] for pair in pairs]
    all_ids = id_target + id_source
    assert len(set(all_ids)) == len(all_ids)

    if add_reals:
        nb_reals = nb_pairs
        real_ids = get_reals(embed, pairs, nb_reals)
    else:
        real_ids = []

    final_df = create_csv(pairs, real_ids)

    final_df.to_csv(path_csv)


if __name__ == '__main__':

    # CelebA
    data_folder = 'dataset/celeba-dataset/versions/2/img_align_celeba/img_align_celeba'
    embed_path = 'dataset/celeba-dataset/versions/2/embeddings.pkl'
    csv_path = 'dataset/celeba-dataset/versions/2/splitted_celeba.csv'

    # CelebAMask-HQ
    data_folder = 'dataset/CelebAMask-HQ/CelebA-HQ-img'
    embed_path = 'dataset/CelebAMask-HQ/embeddings.pkl'
    csv_path = 'dataset/CelebAMask-HQ/splitted_celeba_HQ.csv'


    # extract_embeddings(data_folder, embed_path, nb_imgs=20000)

    split_celeba(
        path_csv=csv_path,
        path_embed=embed_path,
        upper_bound=0.25,
        lower_bound=0.20,
        nb_pairs=1000,
        add_reals=False
    )
