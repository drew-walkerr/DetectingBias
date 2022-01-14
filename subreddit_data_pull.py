import praw
import pandas as pd
import nltk
#Need code to read in client_id, secret, user_agent
api = pd.read_csv("subreddit_data_api.csv",header=None)
#remove header
api.iloc[0,0]
api.iloc[1,0]
api.iloc[2,0]

print(api[0,0])
reddit = praw.Reddit(client_id=api.iloc[0,0],
                     client_secret=api.iloc[1,0],
                     user_agent=api.iloc[2,0])

hot_posts = reddit.subreddit('Suboxone').hot(limit=10)
for post in hot_posts:
    print(post.title)


posts = []
suboxone_subreddit = reddit.subreddit('Suboxone')
for post in suboxone_subreddit.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)