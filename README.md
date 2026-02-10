# Neural Network from Scratch
A minimal neural network implementation from scratch in Python using only NumPy. This project demonstrates the fundamental mechanics of neural networks by solving the XOR problem through backpropagation and gradient descent.

# 🚀 Quick Start
bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch

# Install dependencies
pip install numpy

# Run the neural network
python basic_neural_network.py
📋 Overview
This project implements a basic feedforward neural network with one hidden layer from the ground up. The network is trained on the classic XOR (exclusive OR) problem, which is non-linearly separable and serves as an excellent benchmark for neural network implementations. By avoiding high-level deep learning frameworks, this project provides clear insight into the underlying mathematics and algorithms of neural networks.

<div align="center"> <img src="assets/code.jpg" alt="Code Implementation" width="800"/> <p><em>Complete neural network implementation in Python</em></p> </div>
✨ Features
Feature	Description	Status
🧠 From Scratch	Built without any ML frameworks	✅
⚡ Forward Pass	Complete forward propagation	✅
🔄 Backpropagation	Gradient descent implementation	✅
📊 Loss Monitoring	Mean squared error tracking	✅
🧪 XOR Solution	Solves XOR logic gate problem	✅
🎯 Educational	Clear, commented code for learning	✅
🛠️ Technologies Used
<div align="center">
Technology	Purpose	Logo
Python	Core programming language	<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40" alt="Python">
NumPy	Numerical computing	<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40" height="40" alt="NumPy">
</div>
📁 Project Structure
text
neural-network-from-scratch/
│
├── assets/
│   ├── code.jpg
│   └── output.jpg
│
├── basic_neural_network.py
├── requirements.txt
├── LICENSE
└── README.md
🔧 Installation
Prerequisites
Python 3.8+

pip package manager

Step 1: Clone Repository
bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
Step 2: Create Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install numpy
💻 Usage
Basic Usage
bash
python basic_neural_network.py
Code Example
python
# Network Initialization
nn = NeuralNetwork(
    input_size=2,
    hidden_size=2,
    output_size=1,
    learning_rate=0.5
)

# Training
nn.train(X, y, epochs=10000)

# Testing
output = nn.forward([1, 0])
📈 Results & Visualizations
Program Output
<div align="center"> <img src="assets/output.jpg" alt="Program Output" width="800"/> <p><em>Training progress and final predictions</em></p> </div>
Performance Results
Input	Expected	Predicted
[0, 0]	0	~0.00
[0, 1]	1	~0.99
[1, 0]	1	~0.99
[1, 1]	0	~0.01
Training Progress
text
Epoch 1000, Loss: 0.2543
Epoch 2000, Loss: 0.1324
Epoch 3000, Loss: 0.0812
Epoch 4000, Loss: 0.0421
Epoch 5000, Loss: 0.0254
Epoch 6000, Loss: 0.0178
Epoch 7000, Loss: 0.0135
Epoch 8000, Loss: 0.0108
Epoch 9000, Loss: 0.0089
Epoch 10000, Loss: 0.0057
🤖 How It Works
Architecture
text
Input Layer (2)
        ↓
Hidden Layer (2) + Sigmoid
        ↓
Output Layer (1) + Sigmoid
Algorithm
python
# 1. Forward Propagation
hidden = sigmoid(X · W₁ + b₁)
output = sigmoid(hidden · W₂ + b₂)

# 2. Error Calculation
error = mean_squared_error(y, output)

# 3. Backward Propagation
# Compute gradients
# Update weights

# 4. Repeat
Mathematical Foundation
Sigmoid: σ(x) = 1/(1+e⁻ˣ)

Loss: MSE = 1/n Σ(yᵢ - ŷᵢ)²

Gradient Descent: w = w - η·∇loss

🎯 Customization
Network Parameters
python
# Adjust these values:
nn = NeuralNetwork(
    input_size=2,      # Number of inputs
    hidden_size=4,     # Hidden neurons (try 2, 4, 8)
    output_size=1,     # Number of outputs
    learning_rate=0.3  # Try 0.1, 0.3, 0.5, 0.8
)
Experiment Ideas
Change hidden layer size (2, 4, 8 neurons)

Adjust learning rate (0.1, 0.3, 0.5)

Try different activation functions

Add more hidden layers

Modify training epochs

🤝 Contributing
Contribution Workflow
Fork the Repository

bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
Create Feature Branch

bash
git checkout -b feature/Enhancement
Commit Changes

bash
git commit -m 'Add enhancement'
Push to Branch

bash
git push origin feature/Enhancement
Open Pull Request

Improvement Areas
Multiple hidden layers

Different activation functions

Batch training support

Training visualizations

Momentum optimization

📄 License
This project is licensed under the MIT License - see LICENSE for details.

📞 Contact
<div align="center">
Sayab Arshad Soduzai
AI & Machine Learning Developer

https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white
https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white
https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white

Project Link: https://github.com/yourusername/neural-network-from-scratch

</div>
<div align="center">
If you find this project helpful, please consider giving it a ⭐ on GitHub!

Happy Learning! 🚀

</div>

Gradient descent optimization

Activation functions (sigmoid)

Loss computation (mean squared error)
