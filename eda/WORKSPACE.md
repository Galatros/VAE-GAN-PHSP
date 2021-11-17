## Reading hdf5 files:
a) hdf5 viewer
sudo dpkg --configure -a
sudo apt-get install hdfvie
hdfview

## Install from requirements.txt
conda list -e > requirements.txt
conda create --name pite-v2  --file requirements.txt

conda env export > environment.yml
conda env create -f environment.yml
