
# Simple chessboard checkers segmentation
![Alt text](resources/readme/all.gif?raw=true "hough_line_transform")


## Introduction
The algorithm is designed to segment the checkers on a standard 16-inch competition chessboard. 
The test dataset includes pictures of the board taken under slightly different angles and with different, albeit good lighting

Checker segmentation is based only on the classic computer vision and clustering algorithms.

## Output
The result of the algorithm are points representing the coordinates of successive intersections of checkers on the chessboard.
Cropping function has been implemented, shape of cropped fields is given as the argument of the function.
![Alt text](resources/readme/cropped.gif?raw=true "hough_line_transform_filtered_clustered")

## Segmentation of the checkers

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



