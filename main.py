#  Import libraries required
from glob import glob
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy
from sklearn.metrics import mean_squared_error

class NetflixRatingPrediction:

# Initialising a class
	def __init__(self):
		self.pivot_table = None
		self.pivot_table_values = None
		self.whole_dataset = None
		self.reduced_dataset = None

# Function to get the movie id from filenames of different movie
	def get_movieIdfromFilename(self, string):
		return string.replace("download/training_set\\mv_", "").lstrip("0").split(".txt")[0]

# Function to find the user movie vector of a specific user from their id
	def user_matrix_from_id(self, matrix, id_):
		return matrix.loc[id_].values

# Function to check if a user id exists in the dataset
	def if_user_exists(self, matrix, id_):
		index_ = matrix.index
		return index_.contains(id_)

# Function to calculate cosine similarity between two arrays
	def cosine_similarity(self, a,b):
		A = numpy.array([a,b])
		similarity = numpy.dot(A, A.T)
		square_mag = numpy.diag(similarity)
		inv_square_mag = 1 / square_mag
		inv_square_mag[numpy.isinf(inv_square_mag)] = 0
		inv_mag = numpy.sqrt(inv_square_mag)
		cosine = similarity * inv_mag
		cosine = cosine.T * inv_mag
		return float("{:.4f}".format(cosine[0][1]))

# Function to calculate K-Neighbours 
# Loop through all the user movie vector  and calculate their cosine similarity with the specified user movie vector
	def k_neighbors(self, matrix, index_, X, return_similarities = False):	
		temp = {}
		for index, i in enumerate(tqdm(matrix)):
			for row_index, row in enumerate(X):
				similarity_ = self.cosine_similarity(i, row)
				if not row_index in temp:
					temp[row_index] = [[index_[index], similarity_]]
				else:
					temp[row_index].append([index_[index], similarity_])
		perma = []
		for key_ in temp.keys():
			sorted_ = sorted(temp[key_], key = lambda x: x[1], reverse = True)[:50]
			perma.append(sorted_)
		if return_similarities == True:
			return perma
		else:
			return [[neighbor[0] for neighbor in row] for row in perma]	

# Function to shorten the dataset. We calculate the .8 quantile of the minimum number of ratings in movies and shorten the dataset keep only those elements
	def shorten_dataset(self, df):
		df_movie_summary = df.groupby('MovieId')['Rating'].agg(['count', 'mean'])
		df_movie_summary.index = df_movie_summary.index.map(int)
		movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
		drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
		print('Movie minimum times of review: {}'.format(movie_benchmark))

# Function to shorten the dataset by removing non-active users
		df_cust_summary = df.groupby('User')['Rating'].agg(['count', 'mean'])
		df_cust_summary.index = df_cust_summary.index.map(int)
		cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
		drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index
		print('Customer minimum times of review: {}'.format(cust_benchmark))

		print('Original Shape: {}'.format(df.shape))
		df = df[~df['MovieId'].isin(drop_movie_list)]
		df = df[~df['User'].isin(drop_cust_list)]
		print('After Trim Shape: {}'.format(df.shape))
		return df

# Predict rating given a user_id and movie_id
	def predict_ratings(self, movieids, userids, neighbors):
		user_matrix_array = [self.user_matrix_from_id(self.pivot_table, user) for user in userids]
		similar_users = self.k_neighbors(self.pivot_table_values, list(self.pivot_table.index), user_matrix_array)
		ratings = []
		for user, movie_id in zip(similar_users, movieids):
			user = numpy.array(user[1:])
			df_ = None
			df_ = self.shortened_dataset[(self.shortened_dataset['User'].isin(user)) & (self.shortened_dataset['MovieId'] == movie_id) & (self.shortened_dataset['Rating'] != -1)]
			df_['Sorting'] = df_['User'].apply(lambda x: numpy.where(user == x)[0][0])
			df_ = df_.sort_values(by = ['Sorting'])
			df_ = df_['Rating'][:neighbors].mean()
			ratings.append(df_)
		return ratings

# Prepare the dataset in proper form from file form
	def prepare_data(self):
		frames = []
		movie_files = glob("download/training_set/mv_*.txt")
		#print(movie_files)
		for file_ in tqdm(movie_files):
			df = pd.read_csv(file_, skiprows = 1, names = ['User','Rating','Date'])
			df = df.drop('Date', axis = 1)
			df['MovieId'] = self.get_movieIdfromFilename(file_)
			df['MovieId'] = df['MovieId'].astype('int16')
			df['User'] = df['User'].astype('int32')
			df['Rating'] = df['Rating'].astype('int8')
			df = df[['MovieId', 'User', 'Rating']]
			frames.append(df)

		df = None
		df = pd.concat(frames, ignore_index = True, sort = False)
		self.whole_dataset = df.copy()
		df = self.shorten_dataset(df)
		df = df.drop_duplicates(subset = ['User', 'MovieId'], keep = "last")
		self.shortened_dataset = df
		pivot_table = df.pivot(index = 'User', columns = 'MovieId', values = 'Rating').fillna(-1)
		print("User Item matrix formed.")
		self.pivot_table = pivot_table
		self.pivot_table_values = pivot_table.values

# Function to check mse score of 100 ratings
	def check_mse(self):
		predictions = self.predict_ratings(self.shortened_dataset['MovieId'].values[:100], self.shortened_dataset['User'].values[:100], 5)
		actual = self.shortened_dataset['Rating'].values[:100]
		print(len(predictions), len(actual))
		print(predictions)
		print(actual)
		print("MSE Error = ", mean_squared_error(actual, predictions))

predicter = NetflixRatingPrediction()
predicter.prepare_data()
predicter.check_mse()

# To predict rating of an individual user_id and movie_id pair use the following function where 6702 is the movie_id and 376872 is the user_id

#print(predicter.predict_ratings([6702], [376872], 5))

# To predict multiple user and movie id paris use the same format but just add the other values in to it. So if we want to predict the rating for the movie_id = 6702, user_id = 376872 and movie_id = 6702, user_id = 376872 we would use the following way

#print(predicter.predict_ratings([6702, 6702], [917535, 376872], 5))

#Here the number 5 indicates that we are averaging the ratings of the 5 top most similar users who have seen the movie


