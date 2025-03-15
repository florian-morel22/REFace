

ROOT_DIR=~/AI_project/deepfake-project/REFace

###################################################################################
##################################### SETUP #######################################
###################################################################################

setup:
	bash setup.sh

	python download_models.py

download-celebamask-hq:
	mkdir -p datasets/CelebAMask-HQ
	curl -L -o $(ROOT_DIR)/datasets/CelebAMask-HQ/celebamaskhq.zip\
  	https://www.kaggle.com/api/v1/datasets/download/ipythonx/celebamaskhq
	unzip $(ROOT_DIR)/datasets/CelebAMask-HQ/celebamaskhq.zip -d $(ROOT_DIR)/datasets/CelebAMask-HQ
	rm $(ROOT_DIR)/datasets/CelebAMask-HQ/celebamaskhq.zip

download-celeba:
	mkdir -p datasets/CelebA
	curl -L -o $(ROOT_DIR)/datasets/CelebA/celeba.zip\
  	https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
	unzip $(ROOT_DIR)/datasets/CelebA/celeba.zip -d $(ROOT_DIR)/datasets/CelebA
	rm $(ROOT_DIR)/datasets/CelebA/celeba.zip


##################################################################################
############################### DATASET GENERATION ###############################
##################################################################################


CONFIG="models/REFace/configs/project.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"
DDIM_STEPS=50
N_SAMPLES=1

START=0
STOP=2

# CelebAMask-HQ dataset
DATA_FOLDER=datasets/CelebAMask-HQ/CelebA-HQ-img
OUT_DIR=our_work/data/CelebAMask-HQ-output
HF_IDENTIFIER_DATASET=florian-morel22/deepfake-celebAMask-HQ-REFace
SPLITTED_CELEBA_PATH=./datasets/CelebAMask-HQ/splitted_celeba_HQ.csv

# CelebA dataset
DATA_FOLDER=datasets/CelebA/img_align_celeba/img_align_celeba
OUT_DIR=our_work/data/CelebA-output
HF_IDENTIFIER_DATASET=florian-morel22/deepfake-celebAMask-HQ-REFace
SPLITTED_CELEBA_PATH=./datasets/CelebA/splitted_celeba.csv


CMD_inference_datset = python our_work/scripts/inference_dataset.py\
		--outdir $(OUT_DIR)\
		--data_folder $(DATA_FOLDER)\
		--config "${CONFIG}"\
		--ckpt "${CKPT}"\
		--n_samples $(N_SAMPLES)\
		--scale 3.5\
		--ddim_steps $(DDIM_STEPS)\
		--hf_dataset_identifier $(HF_IDENTIFIER_DATASET)\
		--splitted_celeba_path $(SPLITTED_CELEBA_PATH)\
		--start $(START)\
		--stop $(STOP)

generate-dataset:
	srun --pty --time=04:00:00 --partition=ENSTA-l40s --gpus=1 $(CMD_inference_datset)

###################################################################################
##################################### METRICS #####################################
###################################################################################

CMD_metrics = python our_work/scripts/metrics/my_metrics.py

compute-metrics:
	srun --pty --time=00:30:00 --partition=ENSTA-l40s --gpus=1 $(CMD_metrics)