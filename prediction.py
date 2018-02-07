'''
prediction of what book the user will like
'''

import numpy as np
import pandas as pd
from range_comments import df_score,user_num,book_num

#load
sim = np.load('E:\\Xinqiao\\youshu\\data\\sim.npy')
df_user = pd.read_csv('E:\\Xinqiao\\youshu\\data\\user.csv')
df_book = pd.read_csv('E:\\Xinqiao\\youshu\\data\\book.csv')
book_length = len(book_num)

#which user's for prediction
test_user = 837967

#get books commented by that user
grouped = df_score.groupby(df_score['user_num'])
book_commented = grouped.get_group(test_user)
score_commented_by_user = book_commented.score.tolist()
booknum_commented_by_user = book_commented.book_num.tolist()
print('number of books commented by user = ',len(booknum_commented_by_user))
print('books are:')
for num in booknum_commented_by_user:
	print(num,df_book[df_book['book_num'] == num ]['book_name'].tolist(),book_commented[book_commented['book_num'] == num ]['score'].tolist())

#user's rate matrix
rate = np.zeros(book_length)
for book,score in zip(booknum_commented_by_user,score_commented_by_user):
	rate[book_num.index(book)] = score
rate = np.matrix(rate).transpose()

# a : supplement matrix
a = np.copy(rate)
a[a>0] = 1
# a_mul_a : a multiply a
a_mul_a = a.repeat(book_length,axis = 1)
a_mul_a = a_mul_a + a_mul_a.transpose()
a_mul_a[a_mul_a>0] = 1
#sim_realated : sim matrix realtated to the user
sim_related = a_mul_a * sim

#prediction matrix
pred_numerator = np.array(sim_related.dot(rate).transpose())[0]
pred_denominator = np.sum(sim_related,axis = 1)

#put the zero value in pred
pred_denominator[pred_denominator == 0] = 1
pred = pred_numerator/pred_denominator

#prediction dataframe
df_pred = pd.DataFrame()
df_pred['prediction'] = pred 
book_num_commented = [book_num[s] for s in df_pred.index.tolist()]
df_pred['book_num'] = book_num_commented

#delete book commented by user already from prediction dataframe
df_pred = df_pred[~df_pred['book_num'].isin(booknum_commented_by_user)]
#sort by prediction and drop na
df_pred = df_pred.sort_values(by=['prediction'],ascending=False)
df_pred = df_pred.dropna()

# we have book num and predition score, then get book name by book.csv
df_book = df_book[df_book['book_num'].isin(book_num)]
book_num_list = df_pred['book_num'].tolist()
book_name_list = [df_book[df_book['book_num'] == s ]['book_name'].tolist() for s in book_num_list]
df_pred['book_name'] = book_name_list

#reindex
df_pred = df_pred.reindex()

print(df_pred.to_string())
# print(df_pred)


