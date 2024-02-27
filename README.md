# Final Assignment

This repository serves as the starting kit for the 5LSM0 final assignment.
This assignment is a part of the 5LSM0 course. It involves working with the Cityscapes dataset and training a neural network. The assignment contributes to 50% of your final grade.

## Getting Started

### Dependencies

We already created a DockerContainer with all dependencies to run on Snellius, in the run_main.sh file we refer to this container. You don't have to changes anything for this.

### Installing

To get started with this project, you need to clone the repository to Snellius or your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/5LSM0/FinalAssignment.git
```

After cloning the repository, navigate to the project directory:

```bash
cd FinalAssignment
```

### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.

  
- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.
  

- **model.py:** Defines the neural network architecture.

  
- **train.py:** Contains the code for training the neural network.

### Authors

- T.J.M. Jaspers
- C.H.B. Claessens
- C.H.J. Kusters
