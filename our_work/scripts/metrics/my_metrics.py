import os

from expression_metric import ExpressionMetric
from pose_metric import PoseMetric
from datasets import load_dataset


if __name__ == '__main__':

    # hf_identifier = "florian-morel22/deepfake-REFace"
    hf_identifier = "Supervache/deepfake"
    hf_token = os.getenv("HF_TOKEN")
    local_celeba_path = "dataset/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
    local_hopenet_path = "Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl"
    model="e4s"

    # Load dataset
    hf_dataset = load_dataset(hf_identifier, token=hf_token)


    # Initialize metrics

    pos_metric = PoseMetric(
        local_hopenet_path=local_hopenet_path,
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    expression_metric = ExpressionMetric(
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    # Compute metrics

    pose = pos_metric.compute(
        hf_dataset,
        local_celeba_path,
        model
    )

    exp = expression_metric.compute(
        hf_dataset,
        local_celeba_path,
        model
    )

    print(f"(Model:{model}) position metric : ", pose)
    print(f"(Model:{model}) expression metric : ", exp)