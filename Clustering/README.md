# Clustering  

In this section you can find my personal implementation on grouping images based on visual symmetry 


![alt text](https://github.com/Yega-Noragami/Final-Year-Project-Deep-Learning/blob/main/screeshots/Cluster.png?raw=true)


## Cropping
- I have used my custom object detection model trained using Tensorflow Object Detection API and used the saved model to Crop out the section in which out image(dolphin) is present.
- I have updated the shape of bounding boxes to be square, so that we can change the image size later on easily 

![alt text](https://github.com/Yega-Noragami/Final-Year-Project-Deep-Learning/blob/main/screeshots/31.%20Cropping%20images%20using%20detection%20model.png?raw=true)

## Normalise Images 
- In this section users can change the shape of all images into a fixed size 
- By Changing all pictues to the same size we are normalising the images so that number of features in each image remains the same. 


## Dimention Reduction using t-SNE 
- I have utilised t-SNE dimention reduciton technique to reduce the images into 2-D points so that they can be easily viusalized in the X-Y plane 
- After performing dimention reduction you will obtain a ".csv" file. 

![alt text](https://github.com/Yega-Noragami/Final-Year-Project-Deep-Learning/blob/main/screeshots/35.%20HDB%20Scan%20results%20visalisation%20.png?raw=true)
## Unsupervised Clustering 
- In this project we are dealing with an problem of unsupervised clustering. 
- As the final number of clusters are not known, i have used k-means and HDBSCAN clustering algorith.
- K-means allows the user to visualize the final number of clusters using elbow method. 
- In our second approach, we have used HDBSCAN , after visualizing the data, he can select the final number of parametrs and the images will be automatically assigned into folders. 
- HDBSCAN also has an outlier class which will have all images which the clustering alogrithm did not assign to any cluster. 




