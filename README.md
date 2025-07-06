# argeoPET
**PET reconstruction suite for arbitrary geometries**

---

## 🚀 Installation

First, use conda-forge as in:

```bash
conda create argeopet-env -c conda-forge python=3.10 numpy>=1.23.5 scipy>=1.9.3  matplotlib>=3.6.2 scikit-image>=0.20 libparallelproj=1.4.0
```

And then use pip:

```bash
pip install --no-binary :all: git+https://github.com/gschramm/parallelproj.git@v1.4.0

pip install cupy-cuda12x>=11.4.0

pip install plt-wrapper>=0.0.2
```


---

## 🧪 Example Run

```bash
python run_reconstruction.py
```

---

## 📦 Data

Download and merge the following two zips:

- [Download File 1](https://drive.google.com/file/d/12SGINP-vNgCKZIceSaijLQxiPOOWExJl/view?usp=drive_link)
- [Download File 2](https://drive.google.com/file/d/1efqD4PovUZ5zxdaYGBSJes527Lcfxg7I/view?usp=drive_link)


---

## 📌 Notes

- GPU support is required for Cupy