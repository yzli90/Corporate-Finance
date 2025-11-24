# Deep Learning Solution for Dynamic Corporate Finance Models

This repository contains the implementation for **Interview Project Question 4 (Part 1)**. 

It solves the dynamic investment model from **Strebulaev (2012)** with fixed adjustment costs, utilizing the deep learning framework (All-in-One expectation operator) proposed by **Maliar et al. (2021)**.

## 📄 Formal Report

For the detailed literature review, methodology justification, and analysis of results, please refer to the formal report:

👉 **[Report.pdf](./Report.pdf)**

## 📂 Project Structure

* **`main.py`**: Entry point. Handles the training loop, curriculum learning schedule for smoothing, and result visualization.
* **`model.py`**: Defines the Neural Networks (Value & Policy Nets) and the custom training step implementing Equation 15 from Maliar (2021).
* **`economics.py`**: Contains economic primitives (Cobb-Douglas production, adjustment costs, AR(1) shocks).
* **`test_project.py`**: Comprehensive test suite (Unit, Integration, and Validation tests).
* **`results_maliar_final_stable/`**: Directory where generated plots and data are saved.

## 🚀 Getting Started

### Prerequisites
The project requires Python 3.7+ and the following libraries:
* `tensorflow`
* `numpy`
* `matplotlib`
* `pytest` (for testing)

### Installation
```bash
pip install tensorflow numpy matplotlib pytest
