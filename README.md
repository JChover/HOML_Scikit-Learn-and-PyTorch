# Hands-On Machine Learning with Scikit-Learn and PyTorch

> [!NOTE]
> Book Started: 08/02/2026

![logo.png](logo.png)

## Table of Contents

- [Description](#description)
- [Progress](#progress)
  - [Part I: The Fundamentals of Machine Learning](#part-i-the-fundamentals-of-machine-learning)
  - [Part II: Neural Networks and Deep Learning](#part-ii-neural-networks-and-deep-learning)
  - [Appendices](#appendices)
- [Screenshots](#screenshots)
- [Resources](#resources)
- [Learned](#learned)
- [Conclusions](#conclusions) 

## Description
Here I will keep track of all the exercises and contents learned while reading the book "*Hands-On Machine Learning with Scikit-Learn and PyTorch*" by **Aurélien Geron**.
> [!IMPORTANT]
> No content of the book will be available here, this repository is just a place where I will leave my solutions of the exercises and some of the concepts learned along the way.

## Progress
### Part I: The Fundamentals of Machine Learning

<details open>
<summary><b>Chapter 1: The Machine Learning Landscape</b> ✔️</summary>

  - [ ] <b>What Is Machine Learning?</b> 👈
  - [ ] <b>Why Use Machine Learning?</b>
  - [ ] <b>Examples of Applications</b>
  - [ ] <b>Types of Machine Learning Systems</b>
    - [ ] Training Supervision
    - [ ] Batch Versus Online Learning
    - [ ] Instance-Based Versus Model-Based Learning
  - [ ] <b>Main Challenges of Machine Learning</b>
    - [ ] Insufficient Quantity of Training Data
    - [ ] Nonrepresentative Training Data
    - [ ] Poor-Quality Data
    - [ ] Irrelevant Features
    - [ ] Overfitting the Training Data
    - [ ] Underfitting the Training Data
    - [ ] Deployment Issues
    - [ ] Stepping Back
  - [ ] <b>Testing and Validating</b>
    - [ ] Hyperparameter Tuning and Model Selection
    - [ ] Data Mismatch
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 2: End-to-End Machine Learning Project</b></summary>

  - [ ] <b>Working with Real Data</b>
  - [ ] <b>Look at the Big Picture</b>
    - [ ] Frame the Problem
    - [ ] Select a Performance Measure
    - [ ] Check the Assumptions
  - [ ] <b>Get the Data</b>
    - [ ] Running the Code Examples Using Google Colab
    - [ ] Saving Your Code Changes and Your Data
    - [ ] The Power and Danger of Interactivity
    - [ ] Book Code Versus Notebook Code
    - [ ] Download the Data
    - [ ] Take a Quick Look at the Data Structure
    - [ ] Create a Test Set
  - [ ] <b>Explore and Visualize the Data to Gain Insights</b>
    - [ ] Visualizing Geographical Data
    - [ ] Look for Correlations
    - [ ] Experiment with Attribute Combinations
  - [ ] <b>Prepare the Data for Machine Learning Algorithms</b>
    - [ ] Clean the Data
    - [ ] Handling Text and Categorical Attributes
    - [ ] Feature Scaling and Transformation
    - [ ] Custom Transformers
    - [ ] Transformation Pipelines
  - [ ] <b>Select and Train a Model</b>
    - [ ] Train and Evaluate on the Training Set
    - [ ] Better Evaluation Using Cross-Validation
  - [ ] <b>Fine-Tune Your Model</b>
    - [ ] Grid Search
    - [ ] Randomized Search
    - [ ] Ensemble Methods
    - [ ] Analyzing the Best Models and Their Errors
    - [ ] Evaluate Your System on the Test Set
  - [ ] <b>Launch, Monitor, and Maintain Your System</b>
  - [ ] <b>Try It Out!</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 3: Classification</b></summary>

  - [ ] <b>MNIST</b>
  - [ ] <b>Training a Binary Classifier</b>
  - [ ] <b>Performance Measures</b>
    - [ ] Measuring Accuracy Using Cross-Validation
    - [ ] Confusion Matrices
    - [ ] Precision and Recall
    - [ ] The Precision/Recall Trade-Off
    - [ ] The ROC Curve
  - [ ] <b>Multiclass Classification</b>
  - [ ] <b>Error Analysis</b>
  - [ ] <b>Multilabel Classification</b>
  - [ ] <b>Multioutput Classification</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 4: Training Models</b></summary>

  - [ ] <b>Linear Regression</b>
    - [ ] The Normal Equation
    - [ ] Computational Complexity
  - [ ] <b>Gradient Descent</b>
    - [ ] Batch Gradient Descent
    - [ ] Stochastic Gradient Descent
    - [ ] Mini-Batch Gradient Descent
  - [ ] <b>Polynomial Regression</b>
  - [ ] <b>Learning Curves</b>
  - [ ] <b>Regularized Linear Models</b>
    - [ ] Ridge Regression
    - [ ] Lasso Regression
    - [ ] Elastic Net Regression
    - [ ] Early Stopping
  - [ ] <b>Logistic Regression</b>
    - [ ] Estimating Probabilities
    - [ ] Training and Cost Function
    - [ ] Decision Boundaries
    - [ ] Softmax Regression
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 5: Decision Trees</b></summary>

  - [ ] <b>Training and Visualizing a Decision Tree</b>
  - [ ] <b>Making Predictions</b>
  - [ ] <b>Estimating Class Probabilities</b>
  - [ ] <b>The CART Training Algorithm</b>
  - [ ] <b>Computational Complexity</b>
  - [ ] <b>Gini Impurity or Entropy?</b>
  - [ ] <b>Regularization Hyperparameters</b>
  - [ ] <b>Regression</b>
  - [ ] <b>Sensitivity to Axis Orientation</b>
  - [ ] <b>Decision Trees Have a High Variance</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 6: Ensemble Learning and Random Forests</b></summary>

  - [ ] <b>Voting Classifiers</b>
  - [ ] <b>Bagging and Pasting</b>
    - [ ] Bagging and Pasting in Scikit-Learn
    - [ ] Out-of-Bag Evaluation
    - [ ] Random Patches and Random Subspaces
  - [ ] <b>Random Forests</b>
    - [ ] Extra-Trees
    - [ ] Feature Importance
  - [ ] <b>Boosting</b>
    - [ ] AdaBoost
    - [ ] Gradient Boosting
    - [ ] Histogram-Based Gradient Boosting
  - [ ] <b>Stacking</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 7: Dimensionality Reduction</b></summary>

  - [ ] <b>The Curse of Dimensionality</b>
  - [ ] <b>Main Approaches for Dimensionality Reduction</b>
    - [ ] Projection
    - [ ] Manifold Learning
  - [ ] <b>PCA</b>
    - [ ] Preserving the Variance
    - [ ] Principal Components
    - [ ] Projecting Down to d Dimensions
    - [ ] Using Scikit-Learn
    - [ ] Explained Variance Ratio
    - [ ] Choosing the Right Number of Dimensions
    - [ ] PCA for Compression
    - [ ] Randomized PCA
    - [ ] Incremental PCA
  - [ ] <b>Random Projection</b>
  - [ ] <b>LLE</b>
  - [ ] <b>Other Dimensionality Reduction Techniques</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 8: Unsupervised Learning Techniques</b></summary>

  - [ ] <b>Clustering Algorithms: k-means and DBSCAN</b>
    - [ ] k-Means Clustering
    - [ ] Limits of k-Means
    - [ ] Using Clustering for Image Segmentation
    - [ ] Using Clustering for Semi-Supervised Learning
    - [ ] DBSCAN
    - [ ] Other Clustering Algorithms
  - [ ] <b>Gaussian Mixtures</b>
    - [ ] Using Gaussian Mixtures for Anomaly Detection
    - [ ] Selecting the Number of Clusters
    - [ ] Bayesian Gaussian Mixture Models
    - [ ] Other Algorithms for Anomaly and Novelty Detection
  - [ ] <b>Exercises</b>
</details>

### Part II: Neural Networks and Deep Learning

<details>
<summary><b>Chapter 9: Introduction to Artificial Neural Networks</b></summary>

  - [ ] <b>From Biological to Artificial Neurons</b>
    - [ ] Biological Neurons
    - [ ] Logical Computations with Neurons
    - [ ] The Perceptron
    - [ ] The Multilayer Perceptron and Backpropagation
  - [ ] <b>Building and Training MLPs with Scikit-Learn</b>
    - [ ] Regression MLPs
    - [ ] Classification MLPs
  - [ ] <b>Hyperparameter Tuning Guidelines</b>
    - [ ] Number of Hidden Layers
    - [ ] Number of Neurons per Hidden Layer
    - [ ] Learning Rate
    - [ ] Batch Size
    - [ ] Other Hyperparameters
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 10: Building Neural Networks with PyTorch</b></summary>

  - [ ] <b>PyTorch Fundamentals</b>
    - [ ] PyTorch Tensors
    - [ ] Hardware Acceleration
    - [ ] Autograd
  - [ ] <b>Implementing Linear Regression</b>
    - [ ] Linear Regression Using Tensors and Autograd
    - [ ] Linear Regression Using PyTorch’s High-Level API
  - [ ] <b>Implementing a Regression MLP</b>
  - [ ] <b>Implementing Mini-Batch Gradient Descent Using DataLoaders</b>
  - [ ] <b>Model Evaluation</b>
  - [ ] <b>Building Nonsequential Models Using Custom Modules</b>
    - [ ] Building Models with Multiple Inputs
    - [ ] Building Models with Multiple Outputs
  - [ ] <b>Building an Image Classifier with PyTorch</b>
    - [ ] Using TorchVision to Load the Dataset
    - [ ] Building the Classifier
  - [ ] <b>Fine-Tuning Neural Network Hyperparameters with Optuna</b>
  - [ ] <b>Saving and Loading PyTorch Models</b>
  - [ ] <b>Compiling and Optimizing a PyTorch Model</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 11: Training Deep Neural Networks</b></summary>

  - [ ] <b>The Vanishing/Exploding Gradients Problems</b>
    - [ ] Glorot Initialization and He Initialization
    - [ ] Better Activation Functions
    - [ ] Batch Normalization
    - [ ] Layer Normalization
    - [ ] Gradient Clipping
  - [ ] <b>Reusing Pretrained Layers</b>
    - [ ] Transfer Learning with PyTorch
    - [ ] Unsupervised Pretraining
    - [ ] Pretraining on an Auxiliary Task
  - [ ] <b>Faster Optimizers</b>
    - [ ] Momentum
    - [ ] Nesterov Accelerated Gradient
    - [ ] AdaGrad
    - [ ] RMSProp
    - [ ] Adam
    - [ ] AdaMax
    - [ ] NAdam
    - [ ] AdamW
  - [ ] <b>Learning Rate Scheduling</b>
    - [ ] Exponential Scheduling
    - [ ] Cosine Annealing
    - [ ] Performance Scheduling
    - [ ] Warming Up the Learning Rate
    - [ ] Cosine Annealing with Warm Restarts
    - [ ] 1cycle Scheduling
  - [ ] <b>Avoiding Overfitting Through Regularization</b>
    - [ ] ℓ1 and ℓ2 Regularization
    - [ ] Dropout
    - [ ] Monte Carlo Dropout
    - [ ] Max-Norm Regularization
  - [ ] <b>Practical Guidelines</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 12: Deep Computer Vision Using Convolutional Neural Networks</b></summary>

  - [ ] <b>The Architecture of the Visual Cortex</b>
  - [ ] <b>Convolutional Layers</b>
    - [ ] Filters
    - [ ] Stacking Multiple Feature Maps
    - [ ] Implementing Convolutional Layers with PyTorch
  - [ ] <b>Pooling Layers</b>
  - [ ] <b>Implementing Pooling Layers with PyTorch</b>
  - [ ] <b>CNN Architectures</b>
    - [ ] LeNet-5
    - [ ] AlexNet
    - [ ] GoogLeNet
    - [ ] ResNet
    - [ ] Xception
    - [ ] SENet
    - [ ] Other Noteworthy Architectures
    - [ ] Choosing the Right CNN Architecture
    - [ ] GPU RAM Requirements: Inference Versus Training
    - [ ] Reversible Residual Networks (RevNets)
  - [ ] <b>Implementing a ResNet-34 CNN Using PyTorch</b>
  - [ ] <b>Using TorchVision’s Pretrained Models</b>
  - [ ] <b>Pretrained Models for Transfer Learning</b>
  - [ ] <b>Classification and Localization</b>
  - [ ] <b>Object Detection</b>
    - [ ] Fully Convolutional Networks
    - [ ] You Only Look Once
  - [ ] <b>Object Tracking</b>
  - [ ] <b>Semantic Segmentation</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 13: Processing Sequences Using RNNs and CNNs</b></summary>

  - [ ] <b>Recurrent Neurons and Layers</b>
    - [ ] Memory Cells
    - [ ] Input and Output Sequences
  - [ ] <b>Training RNNs</b>
  - [ ] <b>Forecasting a Time Series</b>
    - [ ] The ARMA Model Family
    - [ ] Preparing the Data for Machine Learning Models
    - [ ] Forecasting Using a Linear Model
    - [ ] Forecasting Using a Simple RNN
    - [ ] Forecasting Using a Deep RNN
    - [ ] Forecasting Multivariate Time Series
    - [ ] Forecasting Several Time Steps Ahead
    - [ ] Forecasting Using a Sequence-to-Sequence Model
  - [ ] <b>Handling Long Sequences</b>
    - [ ] Fighting the Unstable Gradients Problem
    - [ ] Tackling the Short-Term Memory Problem
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 14: Natural Language Processing with RNNs and Attention</b></summary>

  - [ ] <b>Generating Shakespearean Text Using a Character RNN</b>
    - [ ] Creating the Training Dataset
    - [ ] Embeddings
    - [ ] Building and Training the Char-RNN Model
    - [ ] Generating Fake Shakespearean Text
  - [ ] <b>Sentiment Analysis Using Hugging Face Libraries</b>
    - [ ] Tokenization Using the Hugging Face Tokenizers Library
    - [ ] Reusing Pretrained Tokenizers
    - [ ] Building and Training a Sentiment Analysis Model
    - [ ] Bidirectional RNNs
    - [ ] Reusing Pretrained Embeddings and Language Models
    - [ ] Task-Specific Classes
    - [ ] The Trainer API
    - [ ] Hugging Face Pipelines
  - [ ] <b>An Encoder-Decoder Network for Neural Machine Translation</b>
  - [ ] <b>Beam Search</b>
  - [ ] <b>Attention Mechanisms</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 15: Transformers for Natural Language Processing and Chatbots</b></summary>

  - [ ] <b>Attention Is All You Need: The Original Transformer Architecture</b>
    - [ ] Positional Encodings
    - [ ] Multi-Head Attention
    - [ ] Building the Rest of the Transformer
  - [ ] <b>Building an English-to-Spanish Transformer</b>
  - [ ] <b>Encoder-Only Transformers for Natural Language Understanding</b>
    - [ ] BERT’s Architecture
    - [ ] BERT Pretraining
    - [ ] BERT Fine-Tuning
    - [ ] Other Encoder-Only Models
  - [ ] <b>Decoder-Only Transformers</b>
    - [ ] GPT-1 Architecture and Generative Pretraining
    - [ ] GPT-2 and Zero-Shot Learning
    - [ ] GPT-3, In-Context Learning, One-Shot Learning, and Few-Shot Learning
    - [ ] Using GPT-2 to Generate Text
    - [ ] Using GPT-2 for Question Answering
    - [ ] Downloading and Running an Even Larger Model: Mistral-7B
  - [ ] <b>Turning a Large Language Model into a Chatbot</b>
    - [ ] Fine-Tuning a Model for Chatting and Following Instructions Using SFT and RLHF
    - [ ] Direct Preference Optimization (DPO)
    - [ ] Fine-Tuning a Model Using the TRL Library
    - [ ] From a Chatbot Model to a Full Chatbot System
    - [ ] Model Context Protocol
    - [ ] Libraries and Tools
  - [ ] <b>Encoder-Decoder Models</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 16: Vision and Multimodal Transformers</b></summary>

  - [ ] <b>Vision Transformers</b>
    - [ ] RNNs with Visual Attention
    - [ ] DETR: A CNN-Transformer Hybrid for Object Detection
    - [ ] The Original ViT
    - [ ] Data-Efficient Image Transformer
    - [ ] Pyramid Vision Transformer for Dense Prediction Tasks
    - [ ] The Swin Transformer: A Fast and Versatile ViT
    - [ ] DINO: Self-Supervised Visual Representation Learning
    - [ ] Other Major Vision Models and Techniques
  - [ ] <b>Multimodal Transformers</b>
    - [ ] VideoBERT: A BERT Variant for Text plus Video
    - [ ] ViLBERT: A Dual-Stream Transformer for Text plus Image
    - [ ] CLIP: A Dual-Encoder Text plus Image Model Trained with Contrastive Pretraining
    - [ ] DALL·E: Generating Images from Text Prompts
    - [ ] Perceiver: Bridging High-Resolution Modalities with Latent Spaces
    - [ ] Perceiver IO: A Flexible Output Mechanism for the Perceiver
    - [ ] Flamingo: Open-Ended Visual Dialogue
    - [ ] BLIP and BLIP-2
  - [ ] <b>Other Multimodal Models</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 17: Speeding Up Transformers</b></summary>
</details>

<details>
<summary><b>Chapter 18: Autoencoders, GANs, and Diffusion Models</b></summary>

  - [ ] <b>Efficient Data Representations</b>
  - [ ] <b>Performing PCA with an Undercomplete Linear Autoencoder</b>
  - [ ] <b>Stacked Autoencoders</b>
    - [ ] Implementing a Stacked Autoencoder Using PyTorch
    - [ ] Visualizing the Reconstructions
    - [ ] Anomaly Detection Using Autoencoders
    - [ ] Visualizing the Fashion MNIST Dataset
    - [ ] Unsupervised Pretraining Using Stacked Autoencoders
    - [ ] Tying Weights
    - [ ] Training One Autoencoder at a Time
  - [ ] <b>Convolutional Autoencoders</b>
  - [ ] <b>Denoising Autoencoders</b>
  - [ ] <b>Sparse Autoencoders</b>
  - [ ] <b>Variational Autoencoders</b>
  - [ ] <b>Generating Fashion MNIST Images</b>
    - [ ] Discrete Variational Autoencoders
  - [ ] <b>Generative Adversarial Networks</b>
    - [ ] The Difficulties of Training GANs
  - [ ] <b>Diffusion Models</b>
  - [ ] <b>Exercises</b>
</details>

<details>
<summary><b>Chapter 19: Reinforcement Learning</b></summary>

  - [ ] <b>What Is Reinforcement Learning?</b>
  - [ ] <b>Policy Gradients</b>
    - [ ] Introduction to the Gymnasium Library
    - [ ] Neural Network Policies
    - [ ] Evaluating Actions: The Credit Assignment Problem
    - [ ] Solving the CartPole Using Policy Gradients
  - [ ] <b>Value-Based Methods</b>
    - [ ] Markov Decision Processes
    - [ ] Temporal Difference Learning
    - [ ] Q-Learning
    - [ ] Exploration Policies
    - [ ] Approximate Q-Learning and Deep Q-Learning
    - [ ] Implementing Deep Q-Learning
    - [ ] DQN Improvements
  - [ ] <b>Actor-Critic Algorithms</b>
  - [ ] <b>Mastering Atari Breakout Using the Stable-Baselines3 PPO Implementation</b>
  - [ ] <b>Overview of Some Popular RL Algorithms</b>
  - [ ] <b>Exercises</b>
  - [ ] <b>Thank You!</b>
</details>

### Appendices

<details>
<summary><b>Appendix A: Autodiff</b></summary>

  - [ ] <b>Manual Differentiation</b>
  - [ ] <b>Finite Difference Approximation</b>
  - [ ] <b>Forward-Mode Autodiff</b>
  - [ ] <b>Reverse-Mode Autodiff</b>
</details>

<details>
<summary><b>Appendix B: Mixed Precision and Quantization</b></summary>

  - [ ] <b>Common Number Representations</b>
  - [ ] <b>Reduced Precision Models</b>
  - [ ] <b>Mixed-Precision Training</b>
  - [ ] <b>Quantization</b>
    - [ ] Linear Quantization
    - [ ] Post-Training Quantization Using torch.ao.quantization
    - [ ] Quantization-Aware Training (QAT)
    - [ ] Quantizing LLMs Using the bitsandbytes Library
  - [ ] <b>Using Pre-Quantized Models</b>
</details>

## Screenshots

> [!TIP]
> Here I will paste some images that I find relevant.



## Resources
- [Aurélien Geron - Book Repository](https://github.com/ageron/handson-mlp)

## Learned
- __Here I will list the things that I learnt reading this book.__ (Chapters 1:3)


## Conclusions
Here I will leave my conclusions once I finish the book.
