# Word2Vec Recommender System

A simple anime recommendation system built using word2vec from gensim, based on the genres and anime names.

### Dataset

Used the dataset available on kaggle to make this. [Link](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

### How I made this

This is based upon this [article](https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/). <br>
 First, I merged the two datasets present in kaggle on the "anime_id" column. <br>

 Then, tokenized the genres column.
 ```python
df["genre_tokens"] = df["genre"].apply(lambda x: x.split(", "))
# converting the tokens to list to be ready to sent to the word2vec model
genre_sentences = df["genre_tokens"].tolist()
```
<br>


