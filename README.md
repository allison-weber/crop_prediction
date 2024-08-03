# Evaluate performance of light-weight crop prediction model compared to SOTA vision transformer

## Background
Crop prediction is an interesting problem because of it's importance, we all need to eat so agricultural planning ultimately will affect all of us, and because of it's challenges, a wide variety of factors affects how well plants grow. Plus, climate change complicates predictions beyond the year-to-year variation in factors like amount of rain and heat. 

This repository contains the work that I completed as part of a team project for a Deep Learning course I took for my master's of analytics program at Georgia Tech. While the rest of my team focused on replicating the architecture presented paper ["MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer"](https://arxiv.org/pdf/2309.09067) by Lin et al. based on their [code repositoty](https://github.com/fudong03/MMST-ViT/tree/main), my piece of the project was to address the downsides of the MMST-ViT model, particularly the model's size (number of parameters) and the time it takes to train.   

## Data
The data for this project is available for download at HuggingFace, [Tiny-CropNet](https://huggingface.co/datasets/fudong03/Tiny-CropNet), which contains the same dataset used in the MMST-ViT paper, and is a subset of the full [CropNet](https://huggingface.co/datasets/CropNet/CropNet) dataset. The dataset is composed of three different modes of data, Sentinel-2 high-resolution satellite images, meteorological data from the High-Resolution Rapid Refresh (HRRR) atmospheric mode, and the county-level crop yield information obtained from the USDA Quick Statistic Website.   

## Approach
To make the most light-weight possible model, I simplified the architecture down to a 2-layer linear neural network (2 fully-connected linear layers, a hidden layer and output layer separated by an activation function). For this model, I had build our own data preparation pipeline for this model, excluding the Sentinel-2 images, see ```data_prep.ipynb```. When doing data checks after joining the meteorological data (features) to the crop yield amounts (label), I realized the result was a very small dataset with less than 1500 samples for each crop (Soybeans:1413, Corn:1386, Winter Wheat: 398, Cotton: 231). To maintain consitency with the rest of the the team, I proceeded with this dataset, but a future improvement could be to train this model on data from all 48 continental states. 

I tuned 5 hyper-parameters: batch size, learning rate, weight decay, hidden dimension size and activation function. Tuning the hyper-parameters was important to achieve a reasonable level of performance given the limited amount of data and the simple structure of the model, with relatively few parameters compared to MMST-ViT. After an initial grid search (time consuming and limited set of option), I created a random search, which resulted in better sets of hyper-parameter values. 

For this project, I used Pytorch and ran the code in Google Colab.

## Results
For this model, I considered 2 dimensions of success, speed and performance. The primary goal was reducing the training from hours/days to minutes. This was achieved and surpassed, with a single training epoch taking less than half a second on a T4 GPU on Google Colab. With 20 epochs (by which time model had plateaued), training took less than 10 seconds. While I did not expect to achieve the same accuracy as the tuned MMST-ViT, I hoped to at least approach the RMSE of the least successful of the benchmark models preseted in the MMST-ViT paper. After tuning the model for each crop, I achieved this for all the crops except Cotton, which was the crop with the fewest samples, see table of RMSE results below. 

RMSE of different models for the 4 differnet crops:

<img width="268" alt="image" src="https://github.com/user-attachments/assets/97dd9236-3baa-41f7-809c-51db5f8f5c25">

Overall, this model generalizes relatively well. For example, soybeans have a difference of only 0.8 bushels per acre between training and test RMSE.Though for cotton, there was a significant difference between the training RMSE and the test RMSE (training RMSE was actually slightly lower than the benchmarks), a sign the model was over-fitting, which makes sense given the relatively few samples. For corn, decreasing the batch size was one of the key hyper-parameters (while learning rate and weight decay seemed more impactful for the other crops) to achieve this level of performance. As shown in the training and test loss curves in figure below the model performs reasonably well. The similarity of the 2 curves indicates that the model might be under-fitting for all the crops except cotton (note that the scale on the cotton curves is far larger). 

Loss curves:

![image](https://github.com/user-attachments/assets/095db749-65a6-43be-9db1-c432da623227)


Overall, for such a simple model, this level of performance was surprising. It doesn't approach the performance of the MMST-ViT, but depending on the application, it could be sufficient given that it is far simpler to implement and faster to run. Additionally, a deeper model (more linear layers) might close the performance gap without adding too much complexity. Given the encouraging results of this approach, future research could focus on simplifying or modifying the ViT architecture to be more light-weight while maintain the prediction accuracy gains of the MMST-ViT model. 





