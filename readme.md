
# Simple chessboard checkers segmentation
![Alt text](resources/readme/all.gif?raw=true "hough_line_transform")


## Introduction
The algorithm is designed to segment the checkers on a standard 16-inch competition chessboard. 
The test dataset includes pictures of the board taken under slightly different angles and with different, albeit good lighting

Checker segmentation is based only on the classic computer vision and clustering algorithms.

## Scripts execution 

<details>
<summary> <b>Installation</b> </summary>
Due to the use of only image operations and unattended clustering algorithms, the GPU is not required. 
To prepare the environment, just install the libraries from requirements.txt.
</details>
<details>
<summary> <b>Running</b> </summary>
Temporarily there is no specific script configuration. An example usage is in the main.py file.
</details>


## Output
The result of the algorithm are points representing the coordinates of successive corners of checkers on the chessboard.
Cropping function has been implemented, shape of cropped fields is given as the argument of the function.
![Alt text](resources/readme/cropped.gif?raw=true "hough_line_transform_filtered_clustered")

## Algorithm steps details

<details>
<summary><b> Segmentation of the checkers </b></summary>


<details>
<summary> Lines detection on a chessboard</summary>


### Lines detection on a chessboard
![Alt text](resources/readme/lines.gif?raw=true "hough_line_transform")
#### General line detection
Line detection is based on Hough Line Transform.
![Alt text](resources/readme/hough_line_transform.jpg?raw=true "hough_line_transform")
#### Lines filtering
The algorithm filters redundant lines which rho and theta values ​​are similar to the already existing lines.
![Alt text](resources/readme/hough_line_transform_filtered.jpg?raw=true "hough_line_transform_filtered")
#### Lines clustering
The lines are clustered due to the angle of inclination. Clustering is done by the DBSCAN algorithm. Outlier lines are removed from lines list.
![Alt text](resources/readme/hough_line_transform_filtered_clustered.jpg?raw=true "hough_line_transform_filtered_clustered")

</details>

<details>
<summary> Checkers corners detection</summary>


### Lines intersection on a chessboard
![Alt text](resources/readme/intersections.gif?raw=true "hough_line_transform")
#### Intersections
All points of intersection between the horizontal and vertical lines are calculated basis on theta i rho of lines.
![Alt text](resources/readme/intersections.jpg?raw=true "hough_line_transform_filtered_clustered")

#### Intersections clustering
The Intersections are clustered due to the position on the Cartesian plane. Clustering is done by the DBSCAN algorithm.
![Alt text](resources/readme/intersections_clusters.jpg?raw=true "hough_line_transform_filtered_clustered")

#### Intersections clusters centroids
Intersection cluster centroids are calculated as the average of all existing points in the cluster.
![Alt text](resources/readme/intersections_centroids.jpg?raw=true "hough_line_transform_filtered_clustered")

</details>

</details>

## Experiments
<details>
<summary><b> Chess key points detection </b></summary>
I am currently experimenting with detecting chess positions using key points. The key points determined by the SIFT algorithm on a previously properly preprocessed image are focused around chess.

![Alt text](resources/readme/keypoints_2_1.gif?raw=true "hough_line_transform_filtered_clustered")

The density of the accumulation of the key points varies depending on the size of the detection field. Increasing the window causes that the given checker is subject to the detection of key points several times.

Unfortunately, so far I have only been able to focus key points on dark chess. White chess key points accumulation is negligible or it is focused on the contours of the chess. 

Very messy implementation of key points extraction can be found on separate branch.
</details>

