from huggingface_hub import hf_hub_download

print(">> Download usefull models :")

hf_hub_download(repo_id="Sanoojan/REFace", filename="last.ckpt", local_dir="./models/REFace/checkpoints")
hf_hub_download(repo_id="Sanoojan/REFace", filename="Other_dependencies/arcface/model_ir_se50.pth", local_dir=".")
hf_hub_download(repo_id="Sanoojan/REFace", filename="Other_dependencies/face_parsing/79999_iter.pth", local_dir=".")
hf_hub_download(repo_id="Sanoojan/REFace", filename="Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat", local_dir=".")
hf_hub_download(repo_id="Sanoojan/REFace", filename="Other_dependencies/face_recon/epoch_latest.pth", local_dir=".")
hf_hub_download(repo_id="Sanoojan/REFace", filename="Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl", local_dir=".")
hf_hub_download(repo_id="Sanoojan/REFace", filename="BFM/BFM_model_front.mat", local_dir="./eval_tool/Deep3DFaceRecon_pytorch_edit")
