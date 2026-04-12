# MA-RFM

This repository contains the implementation and numerical experiments for MA-RFM, an adaptive random feature method for inverse source reconstruction.  The code includes both 2D and 3D experiments, adaptive mesh refinement, source reconstruction, shape detection, and visualization utilities.

## Repository Layout

```text
MA-RFM/
├── main_code/
│   ├── 2d/                 # Shared 2D implementation
│   └── 3d/                 # Shared 3D implementation
├── Ex4.1/                  # 2D baseline experiment
├── Ex4.2/                  # 2D IA-RFM experiments
├── Ex4.3/                  # 2D MA-RFM experiment with a rectangle/circle source
├── Ex4.4/                  # 2D two-source experiment
├── Ex4.5/                  # 2D sensitivity and noise experiments
├── Ex4.6/                  # 2D general-shape/kidney experiment
├── Ex4.7/                  # 3D multi-source experiment
├── 3D_donut/               # 3D torus/donut experiment
└── limited_aperture/       # Limited-aperture experiments
```

The reusable code is organized under `main_code/2d` and `main_code/3d`.  The experiment folders contain Jupyter notebooks and saved numerical results used to reproduce the examples.

## Main Modules

2D modules:

```text
main.py              Adaptive reconstruction driver
adaptive_int.py      Adaptive mesh refinement
matrix_assemble.py   Forward-operator matrix assembly
inverse_solver.py    Regularized inverse solve and L-curve utilities
net_2d.py            2D random feature models
source_eval.py       Source and gradient evaluation
bound_detect.py      Shape detection and boundary processing
generate_data.py     Synthetic data generation
visual.py            Plotting and mesh visualization
```

3D modules:

```text
main.py              Adaptive reconstruction driver
adaptive_int.py      Adaptive mesh refinement
matrix_assemble.py   Forward-operator matrix assembly
inverse_solver.py    Regularized inverse solve and L-curve utilities
net_3d.py            3D random feature models
source_eval.py       Source and gradient evaluation
bound_detect.py      3D shape detection and boundary processing
generate_data.py     Synthetic data generation
visual.py            3D plotting and grid visualization
```

## Environment

The experiments were developed with Python 3.9 and CUDA-enabled GPU acceleration.  A typical environment includes:

```text
python >= 3.9
numpy
scipy
torch
cupy-cuda12x
matplotlib
seaborn
scikit-learn
opencv-python
jupyter
```

Example installation:

```bash
conda create -n ma-rfm python=3.9
conda activate ma-rfm
pip install numpy scipy matplotlib seaborn scikit-learn opencv-python jupyter
pip install torch torchvision torchaudio
pip install cupy-cuda12x
```

Choose the CuPy package that matches your local CUDA version.  For example, use `cupy-cuda11x` instead of `cupy-cuda12x` if your CUDA runtime is based on CUDA 11.

## GPU Notes

Most notebooks define both a PyTorch device and a CuPy device, for example:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cupy_device = 0
```

These two settings should refer to the same GPU.  If `device` is `cuda:1`, set `cupy_device = 1`.

## Running the Experiments

The experiments are organized as notebooks.  A typical workflow is:

1. Open the notebook in the desired example folder, such as `Ex4.5/Noise_MA_RFM_5%.ipynb` or `3D_donut/MA_RFM_5%.ipynb`.
2. Make sure the notebook appends the correct module path, for example `../main_code/2d` or `../main_code/3d`.
3. Select the correct `device` and `cupy_device`.
4. Run the data-generation cells if the required `.npy` data are not already present.
5. Run the IA-RFM or MA-RFM reconstruction cells.
6. Use the visualization notebook or visualization cells to reproduce the figures.

Important entry points:

```text
Ex4.3/Noise_MA_RFM_5%.ipynb       2D rectangle/circle noisy MA-RFM example
Ex4.4/Noise_MA_RFM_5%.ipynb       2D two-source noisy MA-RFM example
Ex4.5/senstivity_5%.ipynb         Parameter sensitivity study
Ex4.6/MA_RFM_5%.ipynb             2D general-shape/kidney example
Ex4.7/MA_RFM_5%.ipynb             3D multi-source example
3D_donut/MA_RFM_5%.ipynb          3D torus/donut example
```

## Saved Results

Some experiment folders include saved arrays, trained model weights, figures, and intermediate reconstruction data, such as:

```text
*.npy       numerical arrays and reconstruction results
*.pth       saved PyTorch model weights
*.pkl       saved adaptive mesh objects or parameter dictionaries
*.pdf/png   generated figures
```

These files are included for reproducibility and for quickly regenerating figures without rerunning all expensive computations.

## Reproducibility Notes

Several notebooks set random seeds through NumPy and PyTorch.  Exact results may still vary across GPU models, CUDA versions, PyTorch/CuPy versions, and the selected regularization parameter.  The L-curve regularization parameter is selected manually in the provided experiments.

## Citation

If you use this code, please cite the associated MA-RFM paper.  Add the BibTeX entry here after the paper metadata is finalized.

## License

Add a license file before public release if this repository is distributed publicly.
