# GAN-Image-Generation
Implementation of two models. First, a Generative Adversial Network (GAN) is trained to generate faces based on random numbers. Second, an Auxiliary Classifier GAN (ACGAN) is trained to generate faces with certain attributes (in this case: smiling).

### Results
![alt text](https://github.com/kipu1231/GAN-Image-Generation/blob/master/Results/fig_gan.jpg)

# Usage

### Dataset
In order to download the used dataset, a shell script is provided and can be used by the following command.

    bash ./get_dataset.sh
    
The shell script will automatically download the dataset and store the data in a folder called `face`. 

### Packages
The project is done with python3.6. For used packages, please refer to the requirments.txt for more details. All packages can be installed with the following command.

    pip3 install -r requirements.txt
    
### Training
The models can be trained using the following command. To distinguish the training of GAN and ACGAN, the two following commands can be used.

    python3 train_gan.py
    python3 train_acgan.py

### Testing & Visualization
To test the trained models, the provided script can be run by using the following command. Two plots will be generated and saved in predefined folder as output. 

    bash test_models.sh $1

-   `$1` is the folder to which the output `fig_gan.jpg` and `fig_acgan.jpg` is saved.
