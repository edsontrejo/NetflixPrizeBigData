# Import all libraries
from glob import glob
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
import plotly
import plotly.graph_objs as go

# To get list of all movie files in the training_set directory
movie_files = glob("download/training_set/mv_*.txt")

# To get movie id from the movie files
def get_movieIdfromFilename(string):
	return string.replace("download/training_set\\mv_", "").lstrip("0").split(".txt")[0]

# To read the data from the files and construct a dataset with all of it
frames = []
for file_ in tqdm(movie_files):
	df = pd.read_csv(file_, skiprows = 1, names = ['User','Rating','Date'])
	df = df.drop('Date', axis = 1)
	df['MovieId'] = get_movieIdfromFilename(file_)
	df['MovieId'] = df['MovieId'].astype('int16')
	df['User'] = df['User'].astype('int32')
	df['Rating'] = df['Rating'].astype('int8')
	df = df[['User', 'MovieId', 'Rating']]
	frames.append(df)

# Reduce dataset buy keeping popular movies and active users
df = None
df = pd.concat(frames, ignore_index = True, sort = True)
df_movie_summary = df.groupby('MovieId')['Rating'].agg(['count', 'mean'])
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('User')['Rating'].agg(['count', 'mean'])
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
print('Customer minimum times of review: {}'.format(cust_benchmark))

print('Original Shape: {}'.format(df.shape))
df = df[~df['MovieId'].isin(drop_movie_list)]
df = df[~df['User'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))

# Convert the normal dataset into movie-user matrix form
df = df.drop_duplicates(subset = ['User', 'MovieId'], keep = "last")
df = df.pivot(index = 'MovieId', columns = 'User', values = 'Rating').fillna(0)

# Perform PCA to reduce the movies vector to 2 floating point numbers so that we can plot into a graph
pca = IncrementalPCA(n_components=2, batch_size = 2)
transformed_matrix = pca.fit_transform(df)

# Perform Kmeans clustering on the dataset
kmeans = KMeans(n_clusters = 4).fit_predict(transformed_matrix)

# Read the movie filenames so that it can be displayed in the graph along with its point in vector space
movie_title_dataframe = pd.read_csv('download/movie_titles.txt', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
movie_titles = movie_title_dataframe[movie_title_dataframe.index.isin(df.index)]

# Make graph in plotly
data = [go.Scatter(
	x = transformed_matrix[:, 0],
	y = transformed_matrix[:, 1],
	text = list(movie_titles['Name']),
	hoverinfo = "text",
	mode = "markers",
	marker = dict(color = kmeans)
	)]

plotly.offline.plot({"data": data, "layout": go.Layout(title = "Movies Cluster")})
