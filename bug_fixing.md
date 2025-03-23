## RUN Demo.sh :

**Error :**
ImportError: /home/ensta/ensta-morel/.conda/envs/REFace/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found

**Solution :**
https://github.com/pybind/pybind11/discussions/3453

> rm /home/ensta/ensta-morel/.conda/envs/REFace/bin/../lib/libstdc++.so.6

> cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/ensta/ensta-more
l/.conda/envs/REFace/bin/../lib

> strings /home/ensta/ensta-morel/.conda/envs/REFace/bin/../lib/libstdc++.so.6 | grep GLIBCXX_3.4.32

GLIBCXX_3.4.32 should be returned after the last command


**Error :**
ModuleNotFoundError: No module named 'flask'

**Solution :**
pip install flask


**Error :**
FileNotFoundError: [Errno 2] No such file or directory: 'models/REFace/checkpoints/saved.ckpt'

**Solution :**
https://huggingface.co/Sanoojan/REFace/blob/main/last.ckpt


**Error :**
FileNotFoundError: [Errno 2] No such file or directory: 'Other_dependencies/arcface/model_ir_se50.pth'

**Solution :**

cd Other_dependencies/arcface

pip install gdown

gdown --id 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn

**Error :**
RuntimeError: Unable to open Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat

**Solution :**

cd Other_dependencies/DLIB_landmark_det/

git clone https://github.com/italojs/facial-landmarks-recognition.git

mv facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat ./shape_predictor_68_face_landmarks.dat

**Error :**
FileNotFoundError: [Errno 2] No such file or directory: 'Other_dependencies/face_parsing/79999_iter.pth'

**Solution :**
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812
