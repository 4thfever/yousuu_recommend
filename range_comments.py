#range comment 
import pandas as pd 

df_score = pd.read_csv('E:\\Xinqiao\\youshu\\data\\scores.csv')

# delete books without sufficient comments
#get comment number
grouped = df_score.groupby(df_score['book_num']).size()
# the comment number threshold 
book_threshold = 100
book_delete = [i for i,s in grouped.iteritems() if s < book_threshold]
df_score = df_score[~df_score.book_num.isin(book_delete)]

#get user num list and book num list
user_num = df_score['user_num'].drop_duplicates().tolist()
book_num = df_score['book_num'].drop_duplicates().tolist()
user_num = sorted(user_num)
book_num = sorted(book_num)
