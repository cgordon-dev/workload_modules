# Add the NVIDIA driver repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update


# Remove previous installation
sudo apt-get purge -y nvidia-container-toolkit nvidia-container-runtime

# Add the repository again (you've already done this, but just to be sure)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker



sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
