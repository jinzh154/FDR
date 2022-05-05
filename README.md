## **Unsupervised Robustness Ranking for Neural Network Classifiers in Operation**
This repository contains the demo implementation and detailed experimental results for paper *Unsupervised Robustness Ranking for Neural Network Classifiers in Operation* 
You can find supplement materials in the following subfolders.

- demo_code: model, pre-generated advserial examples, and demo code
- Results: experiment results of FDR scores and robust accuracy 

We implemented the proposed approach and carried out experiments using Keras 2.2.4 with TensorFlow 1.12.0, PyTorch 1.7.1, and toolbox, i.e., Adversarial Robustness Toolbox (ART).

To run the experiment: 
1. Install [ART toolbox] (https://github.com/Trusted-AI/adversarial-robustness-toolbox).
2. Generate perturbed images by running mnist_lenet_adv.py, or use the pregenerated adversarial examples in ~/FDR/adv/.
3. Test FDR by running mnist_lenet.py.
