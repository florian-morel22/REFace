import os
import torch
import argparse

from dotenv import load_dotenv
load_dotenv()

from our_work.scripts.metrics.expression_metric import ExpressionMetric
from our_work.scripts.metrics.pose_metric import PoseMetric
from our_work.scripts.metrics.id_metrics import IDMetric
from datasets import load_dataset



def main(args):

    # hf_identifier = "florian-morel22/deepfake-REFace"
    if args.hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    else:
        hf_token = args.hf_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        i:{
            'identifier': identifier,
            'dataset': load_dataset(identifier, token=hf_token),
        }
        for i, identifier in enumerate(args.hf_dataset_identifiers)
    }

    # Ensure all the datasets have same swapps, looking at target ids

    inter_targets = set(datasets[0]['dataset']['train']['id_target'])

    for i in range(1, len(datasets)):
        dataset = datasets[i]['dataset']
        inter_targets = inter_targets & set(dataset['train']['id_target'])

    print(f">> {len(inter_targets)} common swapps between the datasets")

    for dataset in datasets.values():
        dataset['dataset']['train'] = dataset['dataset']['train'].filter(lambda elem: elem['id_target'] in inter_targets)


    # Initialize metrics
    pos_metric = PoseMetric(
        local_hopenet_path=args.local_hopenet_path,
        device=device,
        batch_size=20,
        num_workers=1,
    )

    expression_metric = ExpressionMetric(
        device=device,
        batch_size=20,
        num_workers=1,
    )

    id_metric = IDMetric(
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    # Compute metrics

    for dataset in datasets.values():

        hf_dataset = dataset['dataset']
        identifier = dataset['identifier']

        pose = pos_metric.compute(
            hf_dataset,
            args.local_celeba_path,
            args.model
        )

        exp = expression_metric.compute(
            hf_dataset,
            args.local_celeba_path,
            args.model
        )

        id_ = id_metric.compute(
            hf_dataset,
            args.local_celeba_path,
            args.local_celeba_mask_path,
            args.model
        )

        if args.model is not None:  
            print(f"(Model:{args.model}) position metric : ", pose)
            print(f"(Model:{args.model}) expression metric : ", exp)
            print(f"(Model:{args.model}) id metric : ", id_)
        else:
            print(f"(Dataset: {identifier}) position metric : ", pose)
            print(f"(Dataset: {identifier}) expression metric : ", exp)
            print(f"(Dataset: {identifier}) id metric : ", id_)

        print("-----------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute metrics for a given dataset')
    parser.add_argument('hf_dataset_identifiers', nargs="+", type=str, default=["florian-morel22/deepfake-celebAMask-HQ-REFace"], help='Datasets to use for the computation of the metrics')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token')
    parser.add_argument('--local_celeba_path', type=str, default="dataset/celeba-dataset/versions/2/img_align_celeba/img_align_celeba", help='Path to the celeba dataset')
    parser.add_argument('--local_celeba_mask_path', type=str, default="", help='Path to the masks of the celeba dataset')
    parser.add_argument('--local_hopenet_path', type=str, default="Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl", help='Path to the hopenet model')
    parser.add_argument('--model', type=str, default=None, help='model name to filter the dataset')

    args = parser.parse_args()

    main(args)
