# argeoPET
PET reconstruction suite for arbitrary geometries

Example installation:

conda create argeopet-env -c conda-forge python=3.10 numpy>=1.23.5 scipy>=1.9.3  matplotlib>=3.6.2 scikit-image>=0.20 libparallelproj=1.4.0

pip install --no-binary :all: git+https://github.com/gschramm/parallelproj.git@v1.4.0

pip install cupy-cuda12x>=11.4.0

pip install plt-wrapper>=0.0.2

(To be updated so parproj does not need a fixed version)




Example run:

python run_reconstruction.py




Data:

Download from https://drive.google.com/file/d/12SGINP-vNgCKZIceSaijLQxiPOOWExJl/view?usp=drive_link and merge