문서 군집화를 한 뒤, 이를 시각화 하는 튜토리얼입니다. 

시각화에는 t-SNE 알고리즘이 자주 이용이 되곤 합니다. 하지만, t-SNE는 학습 과정에서 만들어지는 **P의 품질에 따라** 임베딩 결과가 확연하게 달라집니다. 또한 이미 문서 군집화를 수행하였다면, **문서 군집화 결과를 모델에 반영**하고 싶지만, unsupervised로 진행되는 학습 과정에 이 정보를 반영하는건 쉽지 않습니다. 

이 튜토리얼에서는 172개 영화의 네이버영화 평점 리뷰를 바탕으로 Doc2Vec을 학습했던 document vectors를 군집화 한 뒤, 시각화 하는 방법을 소개합니다. 

Plot을 함께 봐야 하기 때문에 github의 [tutorial (click)][tutorial]을 살펴보시기 바랍니다. 

[tutorial]: https://github.com/lovit/clustering_visualization/blob/master/tutorial.ipynb