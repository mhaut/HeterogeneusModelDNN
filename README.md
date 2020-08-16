# Heterogeneous Model Parallelism for Deep Neural Networks 
The Code for "Heterogeneous Model Parallelism for Deep Neural Networks ". []
```
S. Moreno-Alvarez, J. M. Haut, M. E. Paoletti and J. A. Rico-Gallego
Heterogeneous Model Parallelism for Deep Neural Networks 
```

<p align="center">
<img src="https://github.com/mhaut/HeterogeneusModelDNN/blob/master/images/figure5.png" width="400">
</p>


## Getting Started
### Installation

```
git clone https://github.com/mhaut/HeterogeneusModelDNN.git
cd HeterogeneusModelDNN
conda create --name hetmesh --file environment.yml
conda activate hetmesh
git clone https://github.com/tensorflow/mesh
cd mesh
git checkout 6227cc2b1b4eb3a5beacfc4fbb0b3dfca13371c6
cd ..
sh patch_mesh_and_install.sh
```

### Run code

```
# With 3 devices
python main_cifar.py --list_speed 70,25,5
python main_mnist.py --list_speed 40,40,20
```
