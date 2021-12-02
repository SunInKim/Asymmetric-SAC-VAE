# Asymmetric-SAC-VAE
This alogorithm learns the desk tidying tasks with Asymmetric SAC+VAE.

The unit tasks for tidying the desk are closing the drawer and closing the laptop.

A random number of objects are placed on the desk so that it can be operated in a variety of environments.

An exxample of each task is as follows.

### Laptop close

figure

### Drawer close

figure


## Structure
In this case, the algorithm uses RGB, depth, and segmentation images in the virtual environment.

The structure of Asymmetric SAC+VAE is as follows.

figure

학습은 다음과 같은 코드를 통해 진행된다.
```p
  python main.py --GUI GUI --cuda 
```

Download the urdf file for training from Asymmetric_SAC_VAE_urdf at the following link.

link for urdf: https://drive.google.com/drive/folders/1vb5gxkhQ9PLQNBXB_dwg47qoHiaw5EOL?usp=sharing
