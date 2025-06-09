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
## Dataset Overview

The **CICIoV2024** dataset is a comprehensive benchmark designed for cybersecurity research in the Internet of Vehicles (IoV) domain. It contains real-world CAN-BUS traffic data collected from a 2019 Ford vehicle under both normal (benign) and attack conditions. The dataset is intended to support the development and evaluation of machine learning-based intrusion detection systems (IDS) for automotive networks.

### Key Features

- **Total Instances:** 1,408,219  
- **Benign Samples:** 1,223,737  
- **Attack Samples:** 184,482  
- **Number of Features:** 11  
- **Attack Types:** 5 (plus benign class)

### Classes

1. **Benign**  
   Normal CAN-BUS traffic without any attack.

2. **Gas-Spoofing**  
   Malicious manipulation of gas pedal sensor data.

3. **RPM-Spoofing**  
   Spoofing of engine RPM values.

4. **Speed-Spoofing**  
   Falsification of vehicle speed data.

5. **Steering Wheel-Spoofing**  
   Tampering with steering wheel angle information.

6. **DoS (Denial of Service)**  
   Flooding the CAN-BUS with malicious messages to disrupt normal operations.

### Data Characteristics

- **Imbalanced Classes:**  
  The dataset is highly imbalanced, with benign traffic making up the majority of samples. Attack types are less frequent but diverse, providing a realistic challenge for IDS research.
- **Real-World Collection:**  
  Data was gathered from an actual vehicle, ensuring authenticity and practical relevance.
- **Feature Set:**  
  Includes 11 features extracted from CAN-BUS messages, suitable for machine learning algorithms.

### Use Cases

- Training and evaluating intrusion detection models for connected vehicles.
- Benchmarking machine learning algorithms on real IoV security data.
- Research on class imbalance solutions and adversarial attack detection.

---

**For more details, refer to the official [CICIoV2024 dataset documentation](https://www.unb.ca/cic/datasets/iov.html) or the repository’s Jupyter notebook for data exploration and preprocessing steps.**
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
## Requirements

To run and experiment with the CICIoV2024 project, ensure your environment meets the following requirements:

**Software Requirements**

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

**Python Libraries**

- xgboost
- pandas
- scikit-learn
- numpy
- pickle (standard library, for model serialization)

**Recommended Installation**

You can install the required Python libraries using pip:

```bash
pip install xgboost pandas scikit-learn numpy
```

**Hardware Recommendations**

- At least 8 GB RAM (for handling the dataset efficiently)
- Sufficient disk space to store the dataset and generated models

**Optional**

- A virtual environment (such as `venv` or `conda`) is recommended to manage dependencies cleanly.

---

**Note:**  
Before running the Jupyter Notebook (`xgboost_2.ipynb`), ensure all dependencies are installed and the dataset is available in the expected location. If you encounter missing package errors, install them using pip as shown above.

For further details on project setup and usage, refer to the [Getting Started](#getting-started) section of this README[6][7].

[1] https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
[2] https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
[3] https://github.com/othneildrew/Best-README-Template
[4] https://github.com/chimurai/requirements/blob/master/README.md
[5] https://bulldogjob.com/readme/how-to-write-a-good-readme-for-your-github-project
[6] https://www.archbee.com/blog/readme-document-elements
[7] https://www.hatica.io/blog/best-practices-for-github-readme/
[8] https://github.com/Call-for-Code/Project-Sample/blob/main/README.md
[9] https://www.youtube.com/watch?v=E6NO0rgFub4
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
