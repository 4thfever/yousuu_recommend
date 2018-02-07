'''
create books' similarity matrix
algorithm from
'ItemBased Collaborative Filtering Recommendation Algorithms
Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl
GroupLens Research Group/Army HPC Research Center
Department of Computer Science and Engineering
University of Minnesota, Minneapolis, MN 55455'
'''
import pandas as pd 
import os
import numpy as np
import scipy.sparse as sps
from range_comments import df_score,user_num,book_num

#print length of book and user
book_length = len(book_num)
user_length = len(user_num)
print('user_length =',user_length,'book_length = ',book_length)
# grouped = df_score['score'].groupby(df_score['user_num'])
# df_mean = grouped.mean()

value, row, column = [],[],[]
#scores commented by one user
grouped = df_score.groupby(df_score['user_num'])
for i in range(user_length):
	print(i)
	book_commented = grouped.get_group(user_num[i])
	score_commented_by_user = book_commented.score.tolist()
	book_commented_by_user = book_commented.book_num.tolist()
	book_commented_by_user = [book_num.index(s) for s in book_commented_by_user]
	value.append(score_commented_by_user)
	column.append([i]*len(book_commented))	
	row.append(book_commented_by_user)
value = [item for sublist in value for item in sublist]
row = [item for sublist in row for item in sublist]
column = [item for sublist in column for item in sublist]

#create sparse matrix R (rate) by data we got
R = sps.coo_matrix((value, (row,column)), shape=(book_length,user_length))
#create R_sigma_u, which means average score given by user(not count 0 score)
countings=np.bincount(row)
sums=np.bincount(row,weights=value)
R_sigma_u = sums/countings

#delete useless data
value, row, column = [],[],[]

#create R_minus_Ru, which is all non zero element in R minus average score given by that user(in that row)
# to facilitate operation, we use following code
#create a diagnol matrix whose value is R_sigma_u
#create another matrix whose value is one when R has values at the same location in matrix
# then R - diagnol maxtrix * ones matrix
d = sps.diags(R_sigma_u, 0)
b = R.copy()
b.data = np.ones_like(b.data)
R_minus_Ru = (R - d*b)

#R_minus_Ru_square : square element in R_minus_Ru
#R_minus_Ru_square_sigma_u : sigma u for  R_minus_Ru_square
# R_minus_Ru_square_sigma_u_sqrt : then sqrt operation
R_minus_Ru_square = R_minus_Ru.copy()
R_minus_Ru_square.data **= 2
R_minus_Ru_square_sigma_u = R_minus_Ru_square.sum(1)
R_minus_Ru_square_sigma_u_sqrt = np.sqrt(R_minus_Ru_square_sigma_u)

# create sim matrix by get its numerator and dominator
# matrix opeartion is create by myself based on the essay
sim_numerator = R.dot(R.transpose())
sim_denominator = R_minus_Ru_square_sigma_u_sqrt * R_minus_Ru_square_sigma_u_sqrt.transpose()
sim = sim_numerator/sim_denominator

#delete infinite and Nan part in sim
where_are_NaNs = np.isnan(sim)
sim[where_are_NaNs] = 0

where_are_infs = np.isinf(sim)
sim[where_are_infs] = 0

#fill diagonal to zero, which means set similarity to book itself to zero 
np.fill_diagonal(sim, 0)

#save. if already exists, delete then save
sim_file = 'E:\\Xinqiao\\youshu\\data\\sim.npy'
if os.path.exists(sim_file):
    os.remove(sim_file)
np.save(sim_file, sim)
print('sim matrix saved')
