
<div align="center">

<br><br>
# Digit Recognition
</div>

## Overview

This project focuses on recognizing handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model predicts digits from 0 to 9 with high accuracy.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Gautamhirawat/digitRecognizer.git
```

2. Navigate to the project directory:

```bash
cd digitRecognizer
```

3. Ensure you have the necessary files: `main.py` and `mnist_cnn_model.h5`.

## Usage

To run the digit recognition program, execute the following command in your terminal:

```bash
python main.py
```

This will load the pre-trained CNN model from `mnist_cnn_model.h5` and start the digit recognition process.

## File Descriptions

- **main.py**: Contains the code for loading the CNN model and recognizing digits from input images.
- **mnist_cnn_model.h5**: The pre-trained CNN model file.
- **testing.py**: Contains the code for testing your images . I have given 10 images of digit in this repo.
- **digit.ipynb**: This file just shows how you can run it and the expected resuts

## Results

After running the `main.py` script, the program will display the prediction and accuracy.

> [!NOTE]  
> This model is not highly trained , just about 500 images for better results do Train it via notebook or any other python supported Ide .

## Contact

If you have any questions or suggestions, please feel free to contact:

- **Your Name**: Gautam hirawat
- **Email**: code.gautamhirawat@gmail.com

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


