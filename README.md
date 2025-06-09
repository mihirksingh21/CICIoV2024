# CICIoV2024

## Overview

CICIoV2024 is a repository focused on applying XGBoost machine learning techniques to the CICIoV2024 dataset for intrusion detection in Internet of Vehicles (IoV) environments. The project leverages Jupyter Notebooks for experimentation and includes a trained ensemble model for IoV attack detection.

The CICIoV2024 dataset is a benchmark for cybersecurity research in IoV, containing real CAN-BUS traffic collected from a 2019 Ford vehicle under both benign and attack scenarios. The dataset supports the development and evaluation of machine learning-based intrusion detection systems targeting spoofing and Denial-of-Service (DoS) attacks[3][5].

---

## File Structure

```
CICIoV2024/
│
├── README.md
├── xgboost_2.ipynb
├── final_balanced_ensemble.pkl
```

- **README.md**  
  This file. Provides an overview, usage instructions, and context about the project.

- **xgboost_2.ipynb**  
  Jupyter Notebook containing code for data preprocessing, model training, evaluation, and analysis using the XGBoost algorithm.

- **final_balanced_ensemble.pkl**  
  Serialized (pickled) ensemble model trained on a balanced version of the CICIoV2024 dataset for IoV intrusion detection.

---

## About the Dataset

- **Dataset Name:** CICIoV2024
- **Release Year:** 2024
- **Number of Features:** 11
- **Number of Instances:** 1,408,219 (1,223,737 Benign, 184,482 Attack)
- **Classes (6):**
  - Benign
  - Gas-Spoofing
  - RPM-Spoofing
  - Speed-Spoofing
  - Steering Wheel-Spoofing
  - DoS

The dataset is highly imbalanced, with benign traffic dominating. Attack data includes five distinct types, enabling robust evaluation of detection models[5].

---

## Project Highlights

- **Objective:**  
  Develop and evaluate XGBoost-based models for detecting IoV attacks using real CAN-BUS data.

- **Approach:**  
  - Data preprocessing and balancing (e.g., with Random Under Sampling)
  - Model training and evaluation in Jupyter Notebook
  - Export of trained ensemble for deployment or further analysis

- **Technologies:**  
  - Python
  - Jupyter Notebook
  - XGBoost
  - Pickle (for model serialization)

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mihirksingh21/CICIoV2024.git
   cd CICIoV2024
   ```

2. **Install dependencies:**  
   Ensure you have Python 3.x, Jupyter Notebook, and required libraries (XGBoost, pandas, scikit-learn, etc.).

3. **Run the Notebook:**  
   Open `xgboost_2.ipynb` in Jupyter Notebook to review and execute the model training and evaluation steps.

4. **Model Usage:**  
   The trained ensemble model (`final_balanced_ensemble.pkl`) can be loaded using Python's `pickle` module for inference on new CAN-BUS data.

---

## References

- [CICIoV2024 Dataset Details - UNB][3]
- [Dataset Balancing and Analysis][5]

---

## License

This repository is provided for academic and research purposes. Please refer to the original dataset license for usage restrictions.

---

## Acknowledgements

- CICIoV2024 dataset creators and the University of New Brunswick for providing the foundational data[3].
- XGBoost and open-source Python community.

---

## Contact

For issues or contributions, please open a GitHub issue or contact the repository maintainer.

---

*This project supports research in IoV intrusion detection and aims to strengthen vehicular cybersecurity through machine learning.*

[1] https://github.com/mihirksingh21/CICIoV2024
[2] https://github.com/sali446/CICIoV2024
[3] https://www.unb.ca/cic/datasets/iov-dataset-2024.html
[4] https://github.com/sali446/CICIoV2024/blob/main/Project%20Presentation.pptx
[5] https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/9079/2635
[6] https://github.com/sali446/CICIoV2024/actions
[7] https://www.sciencedirect.com/science/article/pii/S2542660524002920
[8] https://www.sciencedirect.com/science/article/pii/S2542660524001501
[9] https://www.kaggle.com/datasets/pushpakattarde/ciciov2024decimalcsv
[10] https://github.com/sali446
[11] https://paperswithcode.com/dataset/cic-iomt-dataset-2024
[12] https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9079
[13] https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
[14] https://github.com/sirthirrygolooo
[15] https://www.nature.com/articles/s41598-025-94445-9
[16] https://set-science.com/manage/uploads/AICCONF2024_0093/SETSCI_AICCONF2024_0093_003.pdf
[17] https://www.datacamp.com/tutorial/xgboost-in-python
[18] https://www.youtube.com/watch?v=GrJP9FLV3FE
