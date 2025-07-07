# argeoPET
**PET reconstruction suite for arbitrary geometries**

---

## 🚀 Installation

First, use conda-forge as in:

```bash
conda create -n argeopet-env -c conda-forge python=3.10 numpy>=1.23.5 scipy>=1.9.3  matplotlib>=3.6.2 scikit-image>=0.20 libparallelproj=1.4.0
```

And then use pip:

```bash
pip install --no-binary :all: git+https://github.com/gschramm/parallelproj.git@v1.4.0

pip install cupy-cuda12x>=11.4.0

pip install plt-wrapper>=0.0.2
```

This environment should let you run the code once you have cloned/downloaded this repository. Note that this is just an example, you might want to install a different version of Python and/or CuPy (e.g., cupy-cuda11x). You can work without CuPy, but not without NumPy. (For now) the most important is fixing parallelproj/libparallelproj to 1.4.0.

---

## 📦 Data

Download and merge the following two zips:

- [Download File 1](https://drive.google.com/file/d/12SGINP-vNgCKZIceSaijLQxiPOOWExJl/view?usp=drive_link) — Dataset with fewer detections
- [Download File 2](https://drive.google.com/file/d/1efqD4PovUZ5zxdaYGBSJes527Lcfxg7I/view?usp=drive_link) — Dataset with many detections


---

## 🧪 Example Run

```bash
python run_reconstruction.py
```

Note: you can change whether the project runs on CuPy or NumPy in __init__.py.

---

## 📌 Notes

- GPU support is required for CuPy

---

## 🙏 Acknowledgements

- The 100μPET project
- Especially, J. Saidi and M. Vicente from the University of Geneva (Geant4 + Allpix2 simulations)
- The excellent ray tracing implemented in `parallelproj`




