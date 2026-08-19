**Introduction**  
Please note that the best performing configuration has been provided as the default.

**Dependencies**
- Python 3.10.12 
- cmake version 3.22.1
- torch 2.7.1+cu128
- torchvision 0.22.1+cu128
- PIL 11.0.0
- Cv2 4.12.0
- Dlib 20.0.0
- Numpy 2.1.2
- OS: Ubuntu 22.04.5 LTS


**How to run**  
1. Extracting CelebA dataset:
  a. Download the CelebA dataset from the following link: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  b. Extract the contents from the downloaded zip file.

2. Creating the face aligned images:
  a. Update the SOURCE_PATH of where the extracted images are located in face_alignment.py
  b. Update the DESTINATION_PATH for where the aligned images should go in face_alignment.py
  c. Update the wanted PADDING value.
  d. Run the following command: python3 face_alignment.py

3. Splitting the image dataset for all four image configurations (raw, padding=0.25, padding=0.5, and padding=0.75)
  a. Update the SOURCE_PATH of where the extracted images are located in split_dataset.py
  b. Update the TRAIN_PATH, VALIDATE_PATH and TEST_PATH for where the split up dataset should go in split_dataset.py
  c. Run the following command: python3 split_dataset.py


4. Run the facial attribute detection task:
  a. Update the TRAIN_PATH, VALIDATE_PATH and TEST_PATH for where the split up dataset should go in facial_attribute_detection.py
  b. Update the MODEL_NAME for which model you want to test in facial_attribute_detection.py
  c. Run the following command: python3 facial_attribute_detection.py
