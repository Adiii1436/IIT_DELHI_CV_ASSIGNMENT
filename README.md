# Pedestrian Detection Using DINO Model
This project involves training and fine-tuning the DINO object detection model on a pedestrian dataset collected within the IIT Delhi campus. The dataset contains 200 images in COCO format, which have been split into training and validation sets.

## Table of Contents
1. Dataset
2. Setup
3. Running the Code
4. Evaluation and Results

<br>
Fine Tuned Model Weights: https://drive.google.com/file/d/14IwLeka4MJ3fuM4okBZc_8VWilP1k0QP/view?usp=sharing

## Dataset

1. The dataset used in this project contains 200 images annotated in COCO format.
   
   ```python
    import gdown
    
    //download the images
    url = 'https://drive.google.com/file/d/1Ae4OC9uy_7bCy1Wd30gzOK3OPh0Ok92l/view?usp=drive_link'
    output_path = 'dataset.zip'
    gdown.download(url, output_path, quiet=False,fuzzy=True)
    
    //download the annotations
    url = 'https://drive.google.com/file/d/1HVeSR3vo-UA9sd6jOJI5viN5e927gk7Y/view?usp=drive_link'
    output_path = 'annotations.json'
    gdown.download(url, output_path, quiet=False,fuzzy=True)
   ```

3. It has been split as follows: <br>Training set: 160 images <br>Validation set: 40 images


## Setup
To set up the environment and dependencies, follow these steps:

1. Clone the DINO repository:
   ```bash
   git clone https://github.com/IDEA-Research/DINO.git
   ```

2. Install dependencies: Inside the DINO directory, run the following commands:

   ```bash
   pip install -r requirements.txt
   ```

3. Fix dependency issues (if necessary):

- If you encounter issues with yapf, install version 0.40.1:
  ```bash
  pip install yapf==0.40.1
  ```
- If you encounter issues with numpy, install version 1.22.0:
  ```bash
  pip install numpy===1.22.0
  ```
- Compile and install custom operations within the DINO model repository:
  ```python
  python setup.py build install
  ```
  
4. Download the pre-trained DINO-4scale model with ResNet-50 backbone as per the repository instructions.
   ```python
   import gdown

   url = 'https://drive.google.com/file/d/1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_/view?usp=drive_link'
   output_path = 'dino_checkpoint.pth'
   gdown.download(url, output_path, quiet=False,fuzzy=True)
   ```

## Running the code
1. Run evaluations on the validation set using the pre-trained weights:
   ```bash
   %cd DINO
   
   !bash /content/DINO/scripts/DINO_eval.sh /content/COCO_split /content/dino_checkpoint.pth
   ```

2. Create script to fine tune the model:

   ```bash
    import os

    script_dir = '/content/DINO/scripts'
    script_path = os.path.join(script_dir, 'DINO_fine_tune.sh')
    
    os.makedirs(script_dir, exist_ok=True)
    
    script_content = '''#!/bin/bash
    coco_path=$1
    output_dir=$2
    pretrained_checkpoint=$3
    
    python main.py \\
      --output_dir $output_dir \\
      --config_file config/DINO/DINO_4scale.py \\
      --coco_path $coco_path \\
      --pretrain_model_path $pretrained_checkpoint \\
      --options dn_scalar=100 embed_init_tgt=True \\
      dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \\
      dn_box_noise_scale=1.0
    '''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"File '{script_path}' created successfully!")
   ```
   
3. Convert it to unix format:
   ```bash
   !apt-get install dos2unix
   
   !dos2unix /content/DINO/scripts/DINO_fine_tune.sh
   ```

4. Fine Tune the Model:
   ```bash
   !bash /content/DINO/scripts/DINO_fine_tune.sh /content/COCO_split /content/DINO/logs/fine_tuned /content/dino_checkpoint.pth
   ```
5. Visualize the loss graphs generated during fine-tuning.
   ```python
      import matplotlib.pyplot as plt
      import json
      
      log_file_path = '/content/DINO/logs/fine_tuned/logs.txt'  
      
      loss_data = []
      with open(log_file_path, 'r') as log_file:
          for line in log_file:
              loss_data.append(json.loads(line))
      
      epochs = list(range(1, len(loss_data) + 1))
      train_losses = [entry.get('train_loss', None) for entry in loss_data]  
      test_losses = [entry['test_loss'] for entry in loss_data]
      
      plt.figure(figsize=(10, 6))
      plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='--')
      plt.plot(epochs, test_losses, label='Test Loss', color='red')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Train and Test Loss Over Time During Fine-Tuning')
      plt.legend()
      plt.grid(True)
      plt.show()
    ```

   ![download (3)](https://github.com/user-attachments/assets/2aec5c58-5595-4bed-ba40-c0d3dbf2caa4)


## Evaluation and Results
1. Run evaluations on the fine tune model:
   ```bash
   %cd DINO
   
   !bash /content/DINO/scripts/DINO_eval.sh /content/COCO_split /content/DINO/logs/fine_tuned/checkpoint.pth
   ```
2. Visualizing the Results:
     - Original bounding box:
   
       ![download](https://github.com/user-attachments/assets/a55a142d-6d70-4577-839c-163cf0d800b9)
       
     - Fine Tuning Result:
  
       ![download (2)](https://github.com/user-attachments/assets/28453fa7-01cb-47c8-bdfa-8366ccef0119)
