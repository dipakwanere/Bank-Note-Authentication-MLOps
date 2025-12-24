Bank Note Authentication
========================

[Data Source](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data)

This repository contains a machine learning project focused on bank note authentication. The dataset used for this project is sourced from Kaggle and contains features extracted from images of genuine and forged bank notes.




Once your project is set and ready. 

Prepare the dockerfile

```
docker build -t banknote-api .
```
Check the image created

```
docker images
```

See the running containers

```
docker ps
```

Run the docker container

```
docker run -p 5000:5000 banknote-api
```

