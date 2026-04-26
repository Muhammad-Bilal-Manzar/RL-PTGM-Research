# RL-PTGM-Research

# Pre-Training Goal-based Models (PTGM) in PointMaze

This repository contains the implementation of **Pre-Training Goal-based Models (PTGM)**, a hierarchical Reinforcement Learning (RL) framework designed to solve complex, long-horizon tasks with sparse rewards. The framework is tested and evaluated on the `PointMaze_Medium-v3` continuous control environment.

## Project Overview

Standard flat Reinforcement Learning often struggles in environments like mazes where rewards are sparse (only granted at the very end). An agent exploring via random low-level actions (e.g., motor torques) is highly unlikely to stumble upon the goal.

PTGM solves this by adopting a "divide and conquer" hierarchical approach:
1.  **Phase 1 (Offline Pre-training):** Learns the basic dynamics of the world from unlabelled, task-agnostic data.
2.  **Phase 2 (Online RL):** Trains a high-level "Manager" to set strategic, discrete waypoint goals for a low-level "Worker" to execute.

## Architecture & Pipeline

The notebook (`PTGM_PointMaze.ipynb`) is structured into distinct, sequential phases:

### 1. Data Collection
We generate a dataset of random/heuristic transitions within the maze. This data represents basic physical interactions without requiring expert demonstrations of solving the actual task.

### 2. Low-Level Policy 
* Trained via **Behavioral Cloning (BC)** using **Hindsight Relabeling**.
* Instead of learning to solve the maze, it simply learns how to reach specific coordinates (goals) over a short time horizon ($k=20$ steps).
* *Result:* The agent masters basic locomotion. *(Refer to `Low Level goal-conditioned policy - BC loss.png` for training convergence).*

### 3. Goal Space Clustering
* A continuous coordinate space is too difficult for the Manager to learn efficiently.
* We extract states from the dataset and use **K-Means Clustering** to compress the continuous state space into 50 discrete "Goal Clusters".
* *Result:* The Manager now has a simplified, multiple-choice menu of waypoints. *(Refer to `Goal Clustering.png` for a visualization of these discrete anchor points).*

### 4. Goal Prior Model
* A neural network trained to predict which goal cluster naturally follows the current state based on historical data.
* *Result:* Provides "common sense" regularization to prevent the Manager from selecting physically impossible goals. *(Refer to `Goal prior Model - training Loss.png` for training performance).*

### 5. High-Level Policy 
* Trained via **Soft Actor-Critic (SAC)** to maximize the actual environment reward.
* Operates using **Temporal Abstraction**: It selects a goal, waits $k=20$ steps for the Worker to execute it, and then evaluates the result.
* Uses a **KL Divergence Penalty** against the Goal Prior Model to stay grounded in realistic behavior.

## Results & Evaluation

The final phase of the notebook evaluates PTGM against standard RL baselines over 100 evaluation episodes. 

*(Refer to `ptgm_results_final.png` for the performance comparison).*

* **PTGM Success Rate:** **~100%**
    * Achieves sample efficiency, reaching a 50% success rate in just the first 100 episodes.
* **Plain SAC Success Rate:** **0%**
    * Demonstrates the fundamental failure of flat RL in sparse-reward environments (getting stuck in local minima/corners).
* **BC-Finetune:** **~20-30%**
    * Shows that simply fine-tuning a pre-trained model without hierarchical temporal abstraction is insufficient.

**Conclusion:** The combination of offline pre-training, hierarchical temporal abstraction, and KL regularization is strictly necessary to solve this class of problems.

## Getting Started

### Prerequisites
The code is written in Python and uses PyTorch. To run the notebook, install the following dependencies:

```bash
pip install gymnasium gymnasium-robotics mujoco scikit-learn matplotlib numpy torch
```

### Running the Code
1. Clone this repository.
2. Open `PTGM_PointMaze.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells sequentially. 
> **Note:** If running in Colab, restart the runtime after the initial pip install cell to ensure Gymnasium and Mujoco are properly loaded.
