## Model Structure Overview

The main approach involves training two sets of models using TensorFlow to perform classification tasks. Below is the breakdown of the project structure:

### Directories and Their Contents:

- **`images` Directory**:
  - Contains various pictures.
  - Includes test images and images used for the UI interface.

- **`models` Directory**:
  - Houses the two trained model sets.
  - Specifically, the CNN model and the MobileNet model.

- **`results` Directory**:
  - Contains visualized graphs from the training process.
  - Two txt files detail the output from the training process.
  - Two graphs depict the change in accuracy and loss curves for the training and validation datasets of both models.

- **`utils` Directory**:
  - Primarily consists of some files written for testing.
  - Not used practically in this project.

### Key Files:

- **`mainwindow.py`**:
  - The interface file.
  - Mainly completed using PyQt5.
  - Allows for image uploads and predicts the category of the uploaded image.

- **`testmodel.py`**:
  - The testing file.
  - Evaluates the accuracy of the two model sets on the validation dataset.
  - This information can also be gleaned from the output in the `results` txt files.

- **`train_cnn.py`**: Contains the code to train the CNN model.

- **`train_mobilenet.py`**: Contains the code to train the MobileNet model.
