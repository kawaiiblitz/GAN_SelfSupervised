# Applied Deep Learning - Group Coursework, April 2024
### COMP0197(D:2023-T2)(S:2023-AP1)-#002-Group_010
**Group: 010**


### Background
This codebase forms the submission of Group 010 for Applied Deep Learing, coursework 2. It contains code to train and test both a self-supervised model and baseline CNN to segment the Oxford IIIT PET dataset. 

### Key Files
- `main.py`: Entry point for all operations.
- `pretrain.py`: Executes the pre-training routine.
- `finetune.py`: Conducts the fine-tuning process (also used for baseline).
- `models.py`: Stores models for both pre-training and fine-tuning.
- `xxx_methods.py`: Supports various functions.

### Folders
- `pretrained_models`: Stores 3 pre-built models supplied with the submission. Details below.
- `data`: Contains training and testing datasets.
- `output`: Stores output models and statistics.

**IMPORTANT** : Due to space limitations, only a small selection of pre-training images are provided with the submission. The remaining images are available for download from here https://1drv.ms/f/s!AqWPzGA-gRsfh9Y6d0ppG_V3Ktv2qQ?e=qeVDrs They need to be copied to the data/pre-training directory. The resulting directory should be data/pre-training/animals and then folders beneath that for each animal (e.g. antelope, badger etc), 


### Included Models from Pre-training (pretrained_models folder)
- `GAN_jigsaw0_64_200e`: GAN trained for 200 epochs WITHOUT a jigsaw shuffler.
- `GAN_jigsaw4_64_200e`: GAN trained for 200 epochs with a 4-tile jigsaw shuffler between the Generator and Discriminator.
- `GAN_jigsaw9_64_200e`: GAN trained for 200 epochs with a 9-tile jigsaw shuffler between the Generator and Discriminator.

### Use Cases
1. **BASELINE:** Create a baseline model for comparison.
2. **PRE-TRAIN:** Train a pre-training model from scratch for use as input to a fine-training model.
3. **FINE-TUNE A:** Employ a pre-saved, pre-trained model for fine-tuning using the complete training dataset.
4. **FINE-TUNE B:** Use a pre-saved pre-trained model for fine-tuning with a subset percentage of the training dataset.
5. **PRE-TRAIN AND FINE-TUNE:** Execute a full cycle from pre-training to fine-tuning.
6. **TEST:** Evaluate a fine-tuned model's performance on segmentation tasks.

### Conda Environment Details
The codebase requires the Pytorch conda module provided with the coursework, supplemented by the following libraries:
- pandas
- matplotlib
- torchmetrics

```bash
pip install torchmetrics pandas matplotlib
```


## Running the code

### Brief Instructions
1. From a command prompt, navigate to the directory containing `main.py`.
2. Activate the appropriate conda environment.
3. Execute `main.py` with command line parameters tailored for your chosen use case (see below).

### Command Line Parameters for Each Use Case
- **To create a baseline model**
```bash
python main.py --mode baseline
```
- **To create a pre-trained model**
```bash
python main.py --mode pt [--jigsaw True|False] [--tiles <int>]
```
- **To create a fine-tuned model using 100% of training data, using a saved model**
```bash
python main.py --mode ft --model_path <model path>
```
- **To create a fine-tuned model using a certain % of training data using a saved model**
```bash
python main.py --mode ft --model_path <model path> --percent <float>
```
- **To run pre-training and fine-tuning in succession**
```bash
python main.py --mode e2e
```
- **To test a fine-tuned model on segmentation**
```bash
python main.py --mode test --model_path <model path>
```

### Outputs
The code will generate three types of artifacts:
1. **Models:** New models produced are saved to the output/models folder.
2. **Statistics:** Outputs per epoch can be found in the output/stats folder.
3. **Images and Segmentations:** Images ouputted are stored in the output/images folder.

#### Models Naming Convention
- Models are named according to the following convention: `mode_jigsaw_percent.pth`.
  - `mode`: [d, g, ft] - discriminator, generator, finetune.
  - `j0`: no jigsaw.
  - `j9`: jigsaw with 9 tiles.
  - `j4`: jigsaw with 4 tiles.
  - `000p`: percentage of training data used in fine-tuning.


