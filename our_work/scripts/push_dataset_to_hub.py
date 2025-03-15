import os

from PIL import Image
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, concatenate_datasets

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
hf_dataset_identifier = f"florian-morel22/deepfake-REFace"

data = [{
    "id": str, #num_source + num_target + model
    "image": Image.Image,
    "fake": int,
    "model": str,
    "id_source": str,
    "id_target": str,
}]

def update_dataset(
        data: list[dict],
        hf_dataset_identifier: str = hf_dataset_identifier,
        hf_token: str=HF_TOKEN
    ) -> None:

    try:
        from datasets import load_dataset
        existing_dataset = load_dataset(hf_dataset_identifier, token=hf_token)['train']
        existing_id = existing_dataset['id']
        existing_id_source = existing_dataset['id_source']
        existing_id_target = existing_dataset['id_target']
    except:
        print("The dataset does not exist. It will be created.")
        existing_dataset = Dataset.from_dict({
            "id": [],
            "image": [],
            "fake": [],
            "model":[],
            "id_source": [],
            "id_target": []
        })
        existing_id = []
        existing_id_source = []
        existing_id_target = []

    
    # Keep only new samples to ensure unicity of samples
    new_data = []
    for elem in data:
        if elem["id"] not in existing_id:
            new_data.append(elem)

            if elem['id_target'] in existing_id_target and elem['id_target'] != -1:
                print(f">> WANING (id: {elem['id']}) - The TARGET is already used as a TARGET for an other fake image.")
            if elem['id_target'] in existing_id_source and elem['id_target'] != -1:
                print(f">> WANING (id: {elem['id']}) - The TARGET is already used as a SOURCE for an other fake image.")
            if elem['id_source'] in existing_id_source and elem['id_source'] != -1:
                print(f">> WANING (id: {elem['id']}) - The SOURCE is already used as a SOURCE for an other fake image.")
            if elem['id_source'] in existing_id_target and elem['id_source'] != -1:
                print(f">> WANING (id: {elem['id']}) - The SOURCE is already used as a TARGET for an other fake image.")

            existing_id.append(elem['id'])
            existing_id_target.append(elem['id_target'])
            existing_id_source.append(elem['id_source'])

        else:
            print(f">> WANING (id: {elem['id']}) - Sample already exists. It has been by passed.")


    new_dataset = Dataset.from_dict({
        "id": [elem["id"] for elem in new_data],
        "image": [elem["image"] for elem in new_data],
        "fake": [elem["fake"] for elem in new_data],
        "model": [elem["model"] for elem in new_data],
        "id_source": [elem["id_source"] for elem in new_data],
        "id_target": [elem["id_target"] for elem in new_data],
    })


    print("create the updated dataset")
    updated_dataset = concatenate_datasets([existing_dataset, new_dataset])

    print(">>> Send the dataset to the huggingface hub")
    try:
        updated_dataset.push_to_hub(hf_dataset_identifier, token=hf_token, private=False)
    except Exception as e:
        print(f"Error : Fail to send the dataset to the hub ({e})")

def remove_data_by_id(
        ids: list[str], 
        hf_dataset_identifier: str = hf_dataset_identifier,
        hf_token: str=HF_TOKEN
    ) -> None:
    
    try:
        existing_dataset = load_dataset(hf_dataset_identifier, token=hf_token)['train']
    except:
        print("Error Dataset does not exist or token unvalid.")
        return None

    updated_dataset = existing_dataset.filter(lambda elem: elem['id'] not in ids)

    print(">>> Send the dataset to the huggingface hub")
    try:
        updated_dataset.push_to_hub(hf_dataset_identifier, token=hf_token, private=False)
    except Exception as e:
        print(f"Error : Fail to send the dataset to the hub ({e})")

def merge_datasets(
        hf_source_datasets: list[str],
        hf_target_dataset: str,
        hf_token: str=HF_TOKEN
    ) -> None:

    datasets = []
    for hf_dataset in hf_source_datasets:
        try:
            datasets.append(load_dataset(hf_dataset, token=hf_token)['train'])
        except:
            print(f"Error : Fail to load the dataset {hf_dataset}")

    updated_dataset = concatenate_datasets(datasets)

    print(">>> Send the dataset to the huggingface hub")
    try:
        updated_dataset.push_to_hub(hf_target_dataset, token=hf_token, private=False)
    except Exception as e:
        print(f"Error : Fail to send the dataset to the hub ({e})")


if __name__ == '__main__':

    ### TEST ###

    update_dataset([{
        "id": 0,
        "image": Image.new('RGB', (256, 256), (0, 0, 0)),
        "fake": 1,
        "model": "test",
        "id_source": 0,
        "id_target": 4
    }, {
        "id": 1,
        "image": Image.new('RGB', (256, 256), (255, 255, 0)),
        "fake": 1,
        "model": "test",
        "id_source": 0,
        "id_target": 2
    }, {
        "id": 1,
        "image": Image.new('RGB', (256, 256), (255, 255, 255)),
        "fake": 1,
        "model": "test",
        "id_source": 0,
        "id_target": 1
    }],
    hf_dataset_identifier=hf_dataset_identifier,
    hf_token=HF_TOKEN
    )

    remove_data_by_id(
        [1, 2],
        hf_dataset_identifier=hf_dataset_identifier,
        hf_token=HF_TOKEN
    )