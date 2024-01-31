# Master Seminar - Unsupervised Anomaly Segmentation - GANomaly

This repository contains the seminar work to apply the GANomaly model to a Brain MRI dataset

## Installation
- No time , shoot your shot at the requirements.txt file but might be broken
- Check https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image/ganomaly for the key dependencies

## Content
- model/ganomaly:
    ```
    ganomaly
    ┣ custom_torch_model.py - test adpations of the provided codebase, i.e different DCGAN
    ┣ lightning_model.py - pl class
    ┣ loss.py - enc,adv,con loss
    ┣ torch_model.py - optimized code based on the anomalib implementation
    ┣ torch_model_k3.py - adaption tests
    ┗ utils.py
    ```
- Evaluation:
    - ganomaly_eval.py - Eval class tho compute metrics & visualizations 
    - run_eval.ipynb - Notebook that uses the previous functionality
    - poster.ipynb - Additional visualizations for the poster



