# 文本学习(sklearn)
@(机器学习)
```python
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer()
>>> corpus = [
...     'This is the first document.',
...     'This is the second second document.',
...     'And the third one.',
...     'Is this the first document?',
... ]
>>> X = vectorizer.fit_transform(corpus)
>>> print X
  (0, 1)	1
  (0, 2)	1
  (0, 6)	1
  (0, 3)	1
  (0, 8)	1
  (1, 5)	2
  (1, 1)	1
  (1, 6)	1
  (1, 3)	1
  (1, 8)	1
  (2, 4)	1
  (2, 7)	1
  (2, 0)	1
  (2, 6)	1
  (3, 1)	1
  (3, 2)	1
  (3, 6)	1
  (3, 3)	1
  (3, 8)	1
>>> vectorizer.get_feature_names() == (
...     ['and', 'document', 'first', 'is', 'one',
...      'second', 'the', 'third', 'this'])
True
>>> X.toarray()
array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 2, 1, 0, 1],
       [1, 0, 0, 0, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 0, 0, 1, 0, 1]])
>>> vectorizer.get_feature_names()
[u'and', u'document', u'first', u'is', u'one', u'second', u'the', u'third', u'this']

>>> vectorizer.vocabulary_.get('document')
1


# 运用于新的检测
>>> vectorizer.transform(['Something completely is new.']).toarray()
array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])


# 两个
>>> bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

>>> bigram_vectorizer.get_feature_names()
[u'and', u'and the', u'document', u'first', u'first document', u'is', u'is the', u'is this', u'one', u'second', u'second document', u'second second', u'the', u'the first', u'the second', u'the third', u'third', u'third one', u'this', u'this is', u'this the']

>>> analyze = bigram_vectorizer.build_analyzer()
>>> analyze('Bi-grams are cool!') == (
...     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
True
>>> X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
>>> X_2
array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]])


# TF IDF 表达(更注重罕见的单词)
# TfidfVectorizer
# TF 和词袋差不多，出现一次+1 
# IDF 逆向加权，越罕见的单词
>>> corpus = [
...     'This is the first document.',
...     'This is the second second document.',
...     'And the third one.',
...     'Is this the first document?',
... ]
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer()

>>> features_name = vectorizer.get_feature_names()
>>> features_name
[u'and', u'document', u'first', u'is', u'one', u'second', u'the', u'third', u'this']
>>> X = vectorizer.fit_transform(corpus)
>>> X.toarray()
```
![](https://ws4.sinaimg.cn/large/006tNc79gy1frlq6x0lhuj30yq09i0ve.jpg)

第二个`corpus`的`second`的idf值最大为`0.85322574`