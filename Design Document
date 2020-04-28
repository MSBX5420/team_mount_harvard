**Design Phase**

**Team:** Mount Harvard

**Members:** Jenna Beutler, Seth Grossman, Hua Miao, Samuel Statton, Nhi Vo

**INTRODUCTION:**

**Purpose:** To investigate what aspects of a tweet drives higher engagement in tweets related to the Coronavirus pandemic.

**Scope:** We will be performing predictive analysis on a dataset of over 8 million tweets in english, between March 4 and March 28.

**Definitions/ things to know:**

- Engagement: number of likes a tweet receives

**DESIGN DATA**

**Data introduction:** We decided to look into data relating to the current global pandemic and social media. The dataset we chose includes 500,000 original tweets (not retweets) scraped from Twitter each day between March 4 and March 28 of 2020, all of which include one of the following hashtags: #coronavirus, #coronavirusoutbreak, #coronavirusPandemic, #covid19, #covid\_19. In tweets after March 17, the data also scraped tweets that included hashtags #epitwitter and #ihavecorona. The original dataset (after merging each day) includes over 8 million rows, and 22 columns, describing different aspects of each individual tweet, like username and tweet ids. This data is not completely comprehensive due to the large volume of tweets being sent each day.

Our dataset can be found here: [https://www.kaggle.com/smid80/coronavirus-covid19-tweets#2020-03-00%20Coronavirus%20Tweets%20(pre%202020-03-12).CSV](https://www.kaggle.com/smid80/coronavirus-covid19-tweets#2020-03-00%20Coronavirus%20Tweets%20(pre%202020-03-12).CSV)

**Data dictionary:** the following list includes all original 22 columns in the data.

**status\_id** : The ID of the actual Tweet.

**user\_id** : The ID of the user account that Tweeted.

**created\_at** : The date and time of the Tweet.

**screen\_name** : The screen name of the account that Tweeted.

**text** : The text of the Tweet.

**source** : The type of app used.

**reply\_to\_status\_id** : The ID of the Tweet to which this was a reply.

**reply\_to\_user\_id** : The ID of the user to whom this Tweet was a reply.

**reply\_to\_screen\_name** : The screen name of the user to whom this Tweet was a reply.

**is\_quote** : Whether this Tweet is a quote of another Tweet.

**is\_retweet** : Whether this Tweet is a retweet.

**favourites\_count** : The number of favourites this Tweet has received.

**retweet\_count** : The number of times this Tweet has been retweeted.

**country\_code** : The country code of the account that Tweeted.

**place\_full\_name** : The name of the place of the account that Tweeted.

**place\_type** : A description of the type of place corresponding with place\_full\_name.

**followers\_count** : The number of followers of the account that Tweeted.

**friends\_count** : The number of friends of the account that Tweeted.

**account\_lang** : The language of the account that Tweeted.

**account\_created\_at** : The date and time that the account that Tweeted was created.

**verified** : Whether the account that Tweeted is verified.

**lang** : The language of the Tweet.

**Operations performed:**

Dropped the following columns in order to simplify the dataset to variables that we need for analysis:

- Status\_id
- user\_id
- reply\_to\_status\_id
- reply\_to\_user\_id
- reply\_to\_screen\_name
- is\_quote
- is\_retweet
- place\_type
- friends\_count
- account\_lang
- account\_created\_at
- country\_code
- place\_full\_name

Subsetted our tweets by English only.

Converted created\_at column to datetime, renamed to &#39;date.&#39;

One-hot encoded verified column

**Machine Learning:**

Running linear regression on the data to predict the number of favorites a tweet receives based on retweets, followers, verified status, and text features. On the small data set, the training model has a 0.80 r-squared value.

**ARCHITECTURE:**

**Tools used:**

- myBinder: for collaboration on code, and connecting to our Github repo.
- Pyspark: for data operations/analysis.
- AWS: for data processing and storage
- Github: for sharing, organizing code, project documents, etc.

**SOLUTION BREAKDOWN**

**Deployment strategy**

We deployed our Python 3 notebook on an EMR cluster through AWS on our full dataset of over 8 million tweets.
