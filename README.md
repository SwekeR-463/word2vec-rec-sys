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

Training the Word2Vec Model from Gensim -> referred this [article](https://www.analyticsvidhya.com/blog/2023/07/step-by-step-guide-to-word2vec-with-gensim/)
```python
from gensim.models import Word2Vec
w2v_model = Word2Vec(sentences=genre_sentences, vector_size=100, window=5, min_count=1, workers=4)
```
<br>

Generated Recommnedations using this
```python
def recommend_anime_by_genre(genre):
    similar_genres = [g[0] for g in w2v_model.wv.most_similar(genre, topn=5)]
    recommendations = merged_df[merged_df["genre"].str.contains("|".join(similar_genres))]
    return recommendations[["name", "genre"]].drop_duplicates()
```
<br>

Here there was use of *gensim.model.Word2Vec.most_similar* and I referred the [docs](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.most_similar.html). It finds top-N most similar words. 

<br>


### Outputs from my code

![Screenshot 2024-12-25 103726](https://github.com/user-attachments/assets/47c2f784-770e-49d0-8e5f-f42ea38d6e29)

![Screenshot 2024-12-25 103923](https://github.com/user-attachments/assets/f3189028-ad3c-44cf-8bb8-fc3ca2325845)
