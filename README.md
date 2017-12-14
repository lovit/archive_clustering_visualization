문서 군집화를 한 뒤, 이를 시각화 하는 튜토리얼입니다. 

## Scatter plot

시각화에는 t-SNE 알고리즘이 자주 이용이 되곤 합니다. 하지만, t-SNE는 학습 과정에서 만들어지는 **P의 품질에 따라** 임베딩 결과가 확연하게 달라집니다. 또한 이미 문서 군집화를 수행하였다면, **문서 군집화 결과를 모델에 반영**하고 싶지만, unsupervised로 진행되는 학습 과정에 이 정보를 반영하는건 쉽지 않습니다. 

이 튜토리얼에서는 172개 영화의 네이버영화 평점 리뷰를 바탕으로 Doc2Vec을 학습했던 document vectors를 군집화 한 뒤, 시각화 하는 방법을 소개합니다. 

Plot을 함께 봐야 하기 때문에 github의 [tutorial (click)][tutorial]을 살펴보시기 바랍니다. 

[tutorial]: https://github.com/lovit/clustering_visualization/blob/master/tutorial.ipynb

172개 영화는 100차원으로 Doc2Vec을 이용하여 표현되었습니다. 하지만 이를 t-SNE로 그대로 임베딩할 경우, 하나의 군집에 속하는 영화들이 흩어지게 됩니다. 

![snap0](/images/snap0.JPG)

우리는 문서 군집화까지 해뒀습니다. 이 결과를 시각화 과정에서 반영해야 문서 군집화 결과를 잘 설명하는 시각화가 될 것입니다. 하지만 t-SNE의 학습 과정에서 군집화의 결과를 이용하지 않았기 때문에 아래처럼 한 군집에 속하는 문서들이라 하더라도 흩어지게 됩니다. 

![snap1](/images/snap1.JPG)

차라리, 군집화 결과의 centroids를 아래 그림처럼 임베딩해두고, 각 군집에 속한 문서들은 해당 centroid 주변에 뿌려두는 것이 군집화의 결과를 설명하기에 더 적절할 것입니다. 

![snap2](/images/snap2.JPG)

이를 위해서 아래 그림처럼 각 군집의 centroids를 먼저 임베딩하고, 그 점들간의 voronoi 경계를 지키면서, 각 점과 군집중심과의 거리에 비레하도록 데이터 포인트를 뿌려둡니다. 그 과정은 위 링크의 튜토리얼에 적어뒀습니다. 

![snap3](/images/snap3.JPG)


## Centroid pairwise-distance matrix heatmap

군집화 결과의 centroids 간의 pairwise distance matrix 를 heatmap 으로 표현하면 군집의 전반적인 특징을 알 수 있습니다. 동일한 내용의 군집들이라면, 이를 바탕으로 후처리(post-processing)을 할 수도 있습니다. 

	from clustervis import visualize_heatmap
	figure, indices_reordered, segments = visualize_heatmap(
                                                  centroids, 
                                                  sort='dist_pole',
                                                  dist_pole_max_dist=0.5, 
                                                  metric='cosine'
                                              )

위 코드는 Pole clustering 을 이용하여 비슷한 centroids 를 정렬함으로써, 해석 가능한 pairwise distance matrix heatmap 을 그려줍니다. 

![heatmap](/demo_data/centroid_heatmap.png)
