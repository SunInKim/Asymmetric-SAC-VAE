# Asymmetric-SAC-VAE
This alogorithm learns the desk tidying tasks with Asymmetric SAC+VAE.

The unit tasks for tidying the desk are closing the drawer and closing the laptop.

A random number of objects are placed on the desk so that it can be operated in a variety of environments.

An exxample of each task is as follows.

### Laptop close

![re_re_1 laptop_close](https://user-images.githubusercontent.com/50347012/144414866-bd8d0cd3-f95f-4561-9e0b-3f6287b70c99.png)



### Drawer close
![re_re_2 drawer_close](https://user-images.githubusercontent.com/50347012/144414873-7cc1c6b5-47e3-4b01-8525-3cb6fb69232d.png)

## Structure
In this case, the algorithm uses RGB, depth, and segmentation images in the virtual environment.

The structure of Asymmetric SAC+VAE is as follows.

![re_re_3 Asymmetric_SAC_VAE_structure](https://user-images.githubusercontent.com/50347012/144414879-6f69eeea-c7d9-4766-8dcd-141031c1de44.png)

학습은 다음과 같은 코드를 통해 진행된다.
```p
  python main.py --GUI GUI --cuda 
```

Download the urdf file for training from Asymmetric_SAC_VAE_urdf at the following link.

link for urdf: https://drive.google.com/drive/folders/1vb5gxkhQ9PLQNBXB_dwg47qoHiaw5EOL?usp=sharing
