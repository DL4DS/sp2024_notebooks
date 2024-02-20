### 1. Write your code

### 2. Create a *.sh file to send your job to SCC (Example: train.sh)

### 3. Run the following command to send your job to SCC (check the SCC cheat sheet (https://dl4ds.github.io/sp2024/materials/) for more information on the parameters you can use)
```
qsub -pe omp 4 -P ds598 -l gpus=1 train.sh
```
If you take a look at train.sh, you'll see that all the hyperparameter values that I'll need to train my model against are defined, and the combinations of these hyperparameters are also defined. 

Say that you want to train your model against 8 different combinations of hyperparameters. For this you would need to add the 
```
#$ -t 1-8
```
flag to your train.sh file. This tells SCC that you want to create 8 different jobs, each with a different combination of hyperparameters. You would then be able to index into the hyperparameters using the $SGE_TASK_ID variable, and send each set of hyperparameters to SCC.

In a nutshell, through this you send out 8 jobs to SCC, each with a different combination of hyperparameters, at once. Instead of running 8 different jobs one after the other, you can run them all at once. Saves a lot of time.

### 4. Check the status of your job
```
qstat -u your_username
```

### 5. Check the output of your job
Highly recommend creating logs in your program to get a sense of the progress of your job.
(check helpers.py and main.py for an example of how to create logs in your program)
This gives you real-time information about the progress of your job. (Akin to adding print statements and viewing it in your terminal when you run your program locally)

NOTE: If you are trying to run train.sh, set the directories in config.json first. 

