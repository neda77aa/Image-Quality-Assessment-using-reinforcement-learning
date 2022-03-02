# Image Quality Assesment Using Reinforcment Learning

The quality of medical images can strongly influence the clinician’s to perceive the appropriate diagnostic features. However, the definition of quality may differ considering the task performed on each image. To deal with this issue, we have implemented two image quality assessment (IQA) techniques in this project. In the first algorithm, we train an agent using reinforcement learning to determine the image quality based on how amenable it is for a defined task without having quality labels that are manually defined by humans. In this method, the training set is used to optimize the defined task predictor network. At the same time, the agent tries to maximize its accumulated reward by achieving a higher performance on the controller-selected validation set. In the second algorithm, we use meta reinforcement learning (meta-RL) to increase the adaptability of both the IQA agent and the task predictor so that they are less dependent on high-quality, expert-labeled training data. We have used PPO and DDPG algorithms for training the controller network.
The results of the first technique show that removing poor-quality images can achieve 85.71% and 90.82% classification accuracies for PneumoniaMNIST and Echocar- diography datasets, respectively, which are higher than the baseline accuracy. Moreover, the utilized adaptation strategy in the IQA algorithm allows for the IQA agent and task predictor to be adapted using as few as 30% of expert labeled Echocardiography data, which drastically reduces the need for expensive expert labeled data.

![Overview of the IQA](/Results/overview1.png)
![Overview of the Adaptable IQA](/Results/overview2.png)

# Dataset
We have used two sets of data in our project to assess the performance of the IQA and adaptable IQA networks in our experiments: 1- PneumoniaMNIST and 2- Echocardiog- raphy. The details of these datasets are discussed in what follows.

![Mnist Dataset](/Results/mnist_data.png)

The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is a binary-class classification of pneumonia against normal. The source training has been split randomly into training and validation set with a ratio of 9 : 1 i. Additionally, we have used its source validation set as the holdout set. The source images are gray-scale, and their sizes are (384 - 2916) × (127 - 2713). The actual data used in our project has been center-cropped and resized into 1 × 36 × 36. Furthermore, some random images of this dataset were artificially corrupted using pepper and salt noises to create poor-quality pictures. Using these images would probably result in low controller scores. 

![Echo Dataset](/Results/echo_data.png)

As the second dataset, we used the Echocardiography dataset, which contains a to- tal number of 17,015 echo studies from 3,157 unique patients. Depending on the echo transducer position (whether parasternal, apical, subcostal, suprasternal) and the orien- tation of the tomographic plane through the heart (long axis, short axis, four-chamber, five-chamber), different echocardiogram views for the heart exist. Our data contains 14 traditional echo views, which define our classes. These images are categorized into four different groups based on their quality; poor, fair, good, and excellent. The whole dataset was labeled in two attempts. In the first attempt, a cardiologist labeled the dataset, and as a separate attempt, two other senior medical students did the labeling procedure two months later, resulting in 3 sets of labels in total. From now on, we refer to the former labeling attempt as expert labeled data and the latter as non-expert labeled data. The defined task is a multiclass classification that predicts the view of each image of the dataset. Each image is a randomly selected frame of a video with a size of 224 × 224. Additionally, the data has been randomly split into training and validation and holdout set with a ratio of 7:2:1. This data is collected from Vancouver General Hospital Picture Archiving and Communication System (PACS) under the approvals from the institutional Medical Research Ethics Board and the Information Privacy Office. 

# Results
## 1- IQA

![IQA Results](/Results/result1.png)

The results of the experiment on the holdout data are depicted in the figure above. As it is evident, by increasing the rejection ratio, which defines the rejected portion of images with the lowest controller scores,  the classification accuracy increased for both PPO and DDPG algorithms compared to the baseline. Maximum classification accuracies for PneumoniaMNIST/Echocardiography using PPO and DDPG algorithms are 85.71%/90.82% and 85.1%/90.15%, respectively. Here the baseline accuracy is  83.3%/88.13%. These figures also indicate that the PPO algorithm has achieved better classification accuracy and cumulative reward compared to the DDPG algorithm. In the IQA algorithm, the task performance peaked before decreasing as more samples were discarded. This observation indicates although the presented IQA algorithm is effective, there is a trade-off between rejecting images with poor quality and classification accuracy.

## 2- Adaptable IQA

![Adaptable IQA Results](/Results/result2.png)

This figure shows the result of performing the adaptable IQA algorithm on the holdout set of Echocaridgraphy data. We evaluated the models for varying k-values, where k is the ratio of expert-labeled samples used for adaptation (k * 100% samples used). The utilized adaptation strategy in the IQA algorithm allows for the IQA agent and task predictor to be adapted using as few as 30% of expert labeled Echocardiography data. The baseline is the accuracy of the classifier trained with expert labeled data. This figure also shows images with high controller score (selected image) and low controller score.

