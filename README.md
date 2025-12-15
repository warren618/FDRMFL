# ✨ FDRMFL Project Overview ✨

## 🧠 Project Abstract
This study focuses on the feature extraction problem in multi-modal data regression. To address three core challenges in real-world scenarios—limited and non-IID data, effective extraction and fusion of multi-modal information, and susceptibility to catastrophic forgetting in model learning—a task-driven supervised multi-modal federated feature extraction method is proposed. The method integrates multi-modal information extraction and contrastive learning mechanisms, and can adapt to different neural network structures as the latent mapping functions for data of each modality. It supports each client to independently learn low-dimensional representations of multi-modal data, and can flexibly control the degree of retention of effective information about the response variable in the predictive variables within the low-dimensional features through parameter tuning. The multi-constraint learning framework constructed by the method guarantees regression accuracy using Mean Squared Error loss. Through the synergistic effect of mutual information preservation constraint, symmetric Kullback-Leibler divergence constraint, and inter-model contrastive constraint, it achieves the retention of task-related information, the extraction, fusion, and alignment of multi-modal features, and the mitigation of representation drift and catastrophic forgetting in non-IID scenarios, respectively. This ensures that the feature extraction process always centers on improving the performance of downstream regression tasks. Experimental results from simulations and real-world data analysis demonstrate that the proposed method achieves more significant performance improvement on downstream regression tasks compared with classical feature extraction techniques.

## 📦 Repository Contents
- 📁 `code/` – all modeling and experimentation notebooks.
- 📽️ `FDRMFL Pre Beamer.pdf` – slide deck used for interim project presentations.
- 📄 `manuscript.pdf` – current manuscript / technical report draft.

### 📚 `code/` Quick Tour
| Notebook | Purpose |
| --- | --- |
| `VAE1.ipynb` | Baseline FDRMFL pipeline (multi-modal VAE + federated aggregation + classical regressors) evaluated on test split A. |
| `VAE2.ipynb` | Same code structure as `VAE1.ipynb`, rerun with test split B for cross-split validation. |
| `VAE3.ipynb` | Identical workflow evaluated on test split C to check robustness. |
| `模拟函数1-3.ipynb` | Synthetic function suites that generate tri-modal data (image / text / vector) to stress-test robustness against noise and missing modalities. |
| `实验部分 y1 (protein).ipynb`, `y2 (water).ipynb`, `y3 (fat).ipynb` | First real-world dataset notebooks focused on protein, water, and fat indicators respectively, covering cleaning, feature extraction, federated training, and error evaluation. |
| `第二套数据集 moisture/oil/protein/starch.ipynb` | Second batch of lab measurements, split into independent notebooks per indicator for licensing isolation and reproducibility. |

## ⚙️ Environment & Dependencies
- ✅ Recommended stack: Python 3.10 + Conda (GPU optional; notebooks gracefully fall back to CPU).
- 🧩 Core packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `ipywidgets`, `matplotlib`.

### 🚀 Quickstart
```bash
conda create -n fdrmfl python=3.10 -y
conda activate fdrmfl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn tqdm ipywidgets matplotlib jupyterlab
jupyter lab
```
Open any `.ipynb` inside Jupyter Lab/Notebook and run the cells from top to bottom following the inline guidance to reproduce the experiments. ✨

## 📓 Why `.ipynb`?
Notebooks bundle runnable code, inline math, visualizations, and experiment outputs in one living document 📒💡, making it far easier to trace reasoning steps and compare variants than scattering the workflow across plain `.py` files.

