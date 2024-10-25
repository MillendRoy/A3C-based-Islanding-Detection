# A3C Reinforcement Learning for Islanding Detection

This repository contains the implementation of an **Asynchronous Advantage Actor-Critic (A3C)** reinforcement learning model from scratch for islanding detection in a power grid. The model is trained to detect when part of the grid becomes an "island" and continues to operate independently. This repository focuses on implementing a reinforcement learning approach to enhance detection accuracy and speed.

## Table of Contents
- [Project Overview](#project-overview)
- [A3C Algorithm Overview](#a3c-algorithm-overview)
- [Files in the Repository](#files-in-the-repository)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

The project uses an A3C-based reinforcement learning approach to detect islanding in power grids. The main challenge is to identify situations where distributed generation continues to power a location despite disconnection from the grid.

The A3C algorithm is particularly well-suited for this task as it allows asynchronous updates and trains multiple agents in parallel, making the detection process faster and more efficient. This implementation includes:

- Custom-built A3C model for training.
- Efficient data preprocessing and environment setup.
- A system to handle islanding and non-islanding event classification.

## A3C Algorithm Overview

The **Asynchronous Advantage Actor-Critic (A3C)** algorithm is a reinforcement learning algorithm that uses multiple agents interacting with different environments in parallel. Each agent learns both a policy (the "Actor") and a value function (the "Critic"). These agents contribute asynchronously to the global policy, which is then updated periodically. A3C is known for its fast convergence and scalability, especially in environments like this where real-time detection is critical.

Key features of the A3C algorithm:
- **Parallel Learning**: Multiple agents run in parallel and share gradients.
- **Actor-Critic**: Combines policy-based and value-based methods, optimizing both.
- **Advantage Function**: Uses an advantage function to reduce variance in policy updates.
- **Asynchronous Training**: Agents act asynchronously, making the algorithm more robust and efficient.

## Files in the Repository

- **ANN_islanding.py**: Implements the Artificial Neural Network (ANN) used as the policy network within the A3C framework. This script defines the architecture for both the Actor and Critic networks.

- **env.py**: Defines the environment for islanding detection simulation. It simulates various grid scenarios and allows agents to interact with these environments to detect islanding events.

- **main.py**: The main script that launches the A3C training process. It handles asynchronous agent spawning, environment setup, and policy updates.

- **model.py**: Contains the implementation of the A3C model, including policy (Actor) and value (Critic) networks, along with training routines.

- **modify.py**: Handles data preprocessing and normalization tasks required before feeding data to the model.

- **my_optim.py**: Implements the custom optimization algorithms used to train the A3C model efficiently.

- **overall.csv**: The dataset consisting of islanding and non-islanding events used for training the model.

- **testt.py**: Script for evaluating the trained A3C model on new data. It tests the model's ability to classify islanding events.

- **train.py**: The script that manages the training loop for the A3C model, including agent initialization, asynchronous updates, and loss calculations.

## How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/millendroy/your-repo-name.git
   
2.  Install the required dependencies:
     ```bash
      pip install -r requirements.txt
      
3. Prepare the dataset: Ensure that overall.csv is placed in the appropriate directory.

4. Run the A3C training script:
    ```bash
    python main.py

5. After training, evaluate the model's performance on test data:
    ```bash
    python testt.py
    
6. You can tweak hyperparameters (learning rates, discount factors, etc.) in the respective scripts to experiment with performance.

## Technologies Used
- Python 3.8+
- PyTorch (for implementing the neural networks in the A3C model)
- NumPy (for numerical calculations)
- Pandas (for dataset management)
- Matplotlib (for visualizing results)

## Future Work
- Improve the A3C model’s performance by experimenting with different architectures for the Actor and Critic networks.
- Test the A3C model on more complex grid scenarios and larger datasets.
- Implement a real-time detection system integrated with live power grid data.
- Add support for other reinforcement learning algorithms like PPO (Proximal Policy Optimization).
## Contributing

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request and describe your changes.

---

Thank you for checking out the project! Feel free to star the repository and share any feedback.


