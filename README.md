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
![image](https://github.com/user-attachments/assets/dfae3b0a-748e-45b5-8fe5-18e41379bd2c)

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

Certainly! Here’s a sample **README.md** section titled **Output** for your CICIoV2024 repository. This section describes the typical outputs generated by the project, including model files, evaluation metrics, and visualizations.

---

## Output

Running the experiments and notebooks in this repository will generate several key outputs:

### 1. Trained Model

- **File:** `final_balanced_ensemble.pkl`  
  A serialized (pickled) XGBoost ensemble model trained on the balanced CICIoV2024 dataset.  
  - **Usage:** Can be loaded for inference or further analysis using Python’s `pickle` module.

### 2. Evaluation Metrics

- **Location:** Displayed within the Jupyter notebook (`xgboost_2.ipynb`) as output cells.
- **Metrics include:**
  - Accuracy
  - Precision, Recall, F1-score (per class and averaged)
  - Confusion Matrix
  - ROC-AUC (if applicable)

### 3. Visualizations
![image](https://github.com/user-attachments/assets/a553c56a-2660-4301-bde1-ec021022bafc)
![image](https://github.com/user-attachments/assets/b3bb6e8a-d310-4d4f-bb35-156ba573ea5f)

- **Types of Output Graphs:**
  - Confusion Matrix heatmap
  - Feature importance plots (e.g., bar charts from XGBoost)
  - ROC curves (if multi-class ROC is implemented)
  - Class distribution plots (before and after balancing)
- **How to Access:**  
  These are generated and displayed as output cells within the notebook.  
  To extract these images for reports or presentations, see the [Extracting Output Graphs](#extracting-output-graphs) section.

### 4. Logs and Intermediate Files

- **Logs:**  
  Training and evaluation logs are shown in notebook output cells.
- **Intermediate Data:**  
  Any temporary files or processed datasets (if saved) will be in the working directory or as specified in the notebook.

---

### Example Output Snippets

- **Model Loading Example:**
  ```python
  import pickle
  with open('final_balanced_ensemble.pkl', 'rb') as f:
      model = pickle.load(f)
  ```

- **Sample Confusion Matrix Output:**  
  ![Confusion Matrix Example](output_folder/confSample Feature Importance Plot:**  
  ![Feature Importance Example](output_folder/feature Extracting Output Graphs

To save output graphs and images from the notebook:
- Use the [ipynb-image-extract](https://github.com/rohzzn/ipynb-image-extract) tool, or
- Right-click and save images from the notebook outputs manually.

---

**All outputs are intended to support the evaluation and deployment of IoV intrusion detection models based on the CICIoV2024 dataset.**

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
