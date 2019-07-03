LOGIC
++++++

Prediction of Rating
--------------------

To predict ratings when given a userid and movieid we perform the following steps -
1. First all the user files are read from the training directory and a whole dataset is formed with the columns - "Movie Id", "User" and "Rating"
2. First we shorten the dataset and keep only those movies which are popular and those users who are active. This improves the overall size of the dataset so that I can be run. To perform the whole experiment on the whole data is practically not possible without expensive servers. Therefore we have shortened the dataset to a point where we can use it
3. Next the shortened dataset is transfer into a user item matrix. Lets say there is a small dataset as follow -

User | Movie Id | Rating
1    |  4       |   5
2    |  2       |   3
3    |  3       |   4
4    |  4       |   2

Converting this into a user item matrix form will result in the following - 

Movie Id | 4 | 2 | 3 | 4 |
User	 |   |   |   |   |
1	 | 5 | 0 | 0 | 0 |
2	 | 0 | 3 | 0 | 0 |
3	 | 0 | 0 | 4 | 0 |
4	 | 2 | 0 | 0 | 0 |

So basically every user has a vector where data regarding their movie ratings is stored. If they have rated a particular movie then it is stored otherwise it is filled with -1. Google about user item matrix to know more about it

When a userid and moveid is given, we first find the vector of the user from within the dataset then using cosine similarity with all the other user vectors we list the users according to their similarity

Then we select only those users who have rated the particular movie and then we select top n users from that set and average the rating they have given for the particular movie to predic the final rating

Finding cosine similarity of top n users is basically Knn as in K nearest neighbours


Visualization
--------------

For this also we have reduced the dataset and this time we have made two matrix one for users and one for movies so that we cluster both similar users and similar movies
We have used Kmeans for that. Nothing complicated with this

There are two html files in the folder that when opened you will be able to see all the visualizations

MSE Error calculated on 100 datapoints -> MSE Error =  0.8420694444444443
