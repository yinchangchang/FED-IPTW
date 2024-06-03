# Federated Inverse Probability Treatment Weighting for Individual Treatment Effect Estimation

This repository contains the official PyTorch implementation of the following paper:

> **Federated Inverse Probability Treatment Weighting for Individual Treatment Effect Estimation**<br>
> [paper]()
>
> ** Abstract:** * We study individual treatment effect (ITE) estimation in a federated setting, which allows us to harness the decentralized data from multiple hospitals. The goal of ITE estimation is to learn a predictive model from the observational covariates (e.g., a patient's health status) to treatment effect (e.g., the difference in survival probability between applying the treatment or not). Due to the unavoidable confounding bias in the collected data, it is critical to decorrelate the covariates and treatments in learning the model. Inverse Probability Treatment Weighting (IPTW) is a well-known technique for such a purpose, which estimates the conditional probability of treatment given the covariates and uses it to re-weight each training example. Extending IPTW into a federated setting, however, is non-trivial: our key observation is that a well-estimated conditional probability only implies global decorrelation (over hospitals), not local decorrelation (within each hospital). As a result, the ITE estimation model would still suffer confounding bias during local training. To address this, we propose a novel IPTW formulation named FED-IPTW, which enforces both global and local decorrelation between covariates and treatments. At the core of FED-IPTW is the need to estimate the conditional probability of each hospital, leading to an interesting combination of personalized and generic federated learning (FL) --- personalized FL for the conditional probability, followed by generic FL for the predictive model that uses the novel FED-IPTW weights. We validate our approach, which we name FED-IPTW, on both synthetic and real-world datasets and demonstrate its superior performance against state-of-the-art methods. *


# Framework
FED-IPTW introduce a new variable H to capture local bias in each client and decorrelate treatments and covariates at both local and global levels.

<img src="src/framework.PNG" width=80%>
Framework: Framework of FED-IPTW. (A) Federated learning in healthcare. (B) Treatment probability learning to remove local confounding bias. (C) Unbiased factual prediction learning for ITE estimation.


<img src="src/training.PNG" width=80%>
Training pipeline.

# Files Directory
    FED-IPTW
    |
    |--main.py
    |
    |--src
    |    |--loaders							
    |    |    
    |    |--datasets						* dataloader
    |    |    
    |    |--models  						* model loader
    |    |    
    |    |--server  						* server class
    |    |    
    |    |--client  						* client class
    |
    |--data                                 * Put the downloaded datasets here.
    |    |--synthetic
    |    |
    |    |--sepsis
    |
    |--result




# Train FED-IPTW
```
python3 main.py \
            --exp_name "FedAvg_synthetic" --seed 100 --device cuda \
            --dataset synthetic \
            --split_type noniid --test_fraction 0.3 \
            --model_name LogReg \
            --algorithm fedavg --eval_fraction 1 --eval_type global --eval_every 1 --eval_metrics acc1 \
            --R 50 --E 1 --C 1 --B 5 --beta 0 \
            --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss

```
