

ROOT_DIR=~/AI_project/deepfake-project/REFace

CONFIG="models/REFace/configs/project.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"
DDIM_STEPS=50

###################################################################################
##################################### SETUP #######################################
###################################################################################

setup:
	bash setup.sh
	python download_models.py

CELEBAMASK_ANNO_PATH=./datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno
CELEBAMASK_OVERALL_PATH=./datasets/CelebAMask-HQ/CelebA-HQ-mask

download-celebamask-hq:
	mkdir -p datasets/
	curl -L -o $(ROOT_DIR)/datasets/celebamaskhq.zip\
  	https://www.kaggle.com/api/v1/datasets/download/ipythonx/celebamaskhq
	unzip $(ROOT_DIR)/datasets/celebamaskhq.zip -d $(ROOT_DIR)/datasets
	rm $(ROOT_DIR)/datasets/celebamaskhq.zip

	mkdir -p $(CELEBAMASK_OVERALL_PATH)
	python process_CelebA_mask.py $(CELEBAMASK_ANNO_PATH) $(CELEBAMASK_OVERALL_PATH)
	rm -r $(CELEBAMASK_ANNO_PATH)

download-celeba:
	mkdir -p datasets/CelebA
	curl -L -o $(ROOT_DIR)/datasets/CelebA/celeba.zip\
  	https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
	unzip $(ROOT_DIR)/datasets/CelebA/celeba.zip -d $(ROOT_DIR)/datasets/CelebA
	rm $(ROOT_DIR)/datasets/CelebA/celeba.zip


##################################################################################
############################### SIMPLE GENERATION ################################
##################################################################################

SOURCE=our_work/data/input/thomas.jpg
TARGET=our_work/data/input/julian.jpg
OUT_PATH=our_work/data/output/faceswapp.jpg

CMD_generate = python our_work/scripts/inference.py $(SOURCE) $(TARGET) $(OUT_PATH)\
		--config "${CONFIG}"\
		--ckpt "${CKPT}"\
		--ddim_steps $(DDIM_STEPS)\

faceswapp:
	srun --pty --time=04:00:00 --partition=ENSTA-l40s --gpus=1 $(CMD_generate)

headswapp:
	srun --pty --time=04:00:00 --partition=ENSTA-l40s --gpus=1 $(CMD_generate) --head_swapp

##################################################################################
############################### DATASET GENERATION ###############################
##################################################################################


N_SAMPLES=1

# Generate idx START to idx STOP
START=0
STOP=10

#### CelebAMask-HQ dataset ####
DATA_FOLDER=datasets/CelebAMask-HQ/CelebA-HQ-img
OUT_DIR=our_work/data/CelebAMask-HQ-output
HF_IDENTIFIER_DATASET=florian-morel22/deepfake-todelete
SPLITTED_CELEBA_PATH=./datasets/splitted_celeba_HQ.csv

####### CelebA dataset ########
# DATA_FOLDER=datasets/CelebA/img_align_celeba/img_align_celeba
# OUT_DIR=our_work/data/CelebA-output
# HF_IDENTIFIER_DATASET=florian-morel22/deepfake-todelete
# SPLITTED_CELEBA_PATH=./datasets/splitted_celeba.csv


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


METRICS__HF_IDENTIFIER_DATASETS=Supervache/deepfake_celebaHQ florian-morel22/deepfake-celebAMask-HQ-REFace
METRICS__LOCAL_CELEBA_PATH=./datasets/CelebAMask-HQ/CelebA-HQ-img
METRICS__LOCAL_HOPENET_PATH=./Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl
METRICS__LOCAL_CELEBA_MASK_PATH=./datasets/CelebAMask-HQ/CelebA-HQ-mask


CMD_metrics = python our_work/scripts/metrics/my_metrics.py\
		$(METRICS__HF_IDENTIFIER_DATASETS)\
		--local_celeba_path $(METRICS__LOCAL_CELEBA_PATH)\
		--local_celeba_mask_path $(METRICS__LOCAL_CELEBA_MASK_PATH)\
		--local_hopenet_path $(METRICS__LOCAL_HOPENET_PATH)\

compute-metrics:
	srun --pty --time=02:00:00 --partition=ENSTA-l40s --gpus=1 $(CMD_metrics)