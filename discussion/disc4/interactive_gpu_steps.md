## 1. Request for an interactive session on a GPU node
```
qrsh -pe omp 4 -P ds598 -l gpus=1
```
## 2. Go to your '/projectnb/ds598/students/{your_username}' directory
When you request an interactive session, you will be in your home directory. You would need to go to your project directory to access your files.

```
cd /projectnb/ds598/students/xthomas
```
## 3. Load your conda environment
```
module load miniconda
conda activate ds598
```
## 4. Check if the GPU is available
```
nvidia-smi
```
## 5. Run your code
```
python3 gpu_training.py
```

For multi-GPU training,
```
CUDA_VISIBLE_DEVICES=0,1 python3 gpu_training.py
```

## 6. To keep monitoring the GPU usage

Create a screen session to run your code in the background. You can then detach from the screen and use ```nvidia-smi``` to monitor the GPU usage.

Steps to create a screen session:

```
screen -S screen_name
steps 2 - 5
ctrl + a + d (To detach from the screen)
```
After these steps, you would be back to the terminal and you can use ```nvidia-smi``` to monitor the GPU usage.

To go back to the screen session, use the following command:

```
screen -r screen_name
```

To view this on VSCode, right click on the file and select "Open Preview" or  command/ctrl + shift + v on your keyboard.
