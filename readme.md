## TensorFlow 2.3-based Fruit and Vegetable Recognition System
Hello everyone! I'm Lei. This is the code for fruit and vegetable recognition. It also serves as a template for TensorFlow-based object classification. I hope it proves helpful to you.

### Code Structure
We mainly classify by training two models using TensorFlow. The project's directory structure is as follows:

- `images/`: Stores various images, including those for testing and those used in the GUI.
- `models/`: Contains two trained models: the CNN model and the MobileNet model.
- `results/`: Houses visual representations from the training process. There are two .txt files detailing the training outputs, and two graphs showing the training and validation accuracy and loss curves for both models.
- `utils/`: Some auxiliary files I wrote for testing; they don't have practical use in this project.
- `get_data.py`: A script for web scraping images from Baidu.
- `window.py`: The GUI file, crafted with PyQt5. It allows users to upload images and predicts their category.
- `testmodel.py`: A testing script for checking the accuracy of the two models on the validation dataset. This information can also be extracted from the txt files in the `results/` directory.
- `train_cnn.py`: The code for training the CNN model.
- `train_mobilenet.py`: The code for training the MobileNet model.
- `requirements.txt`: Lists the dependencies required for this project.
