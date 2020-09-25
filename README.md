# Hateful Memes

## How do you end up in the top 5% of a U$S100000 competition?

![alt text](https://github.com/SebKleiner/Hateful_Memes/blob/master/submission3.JPG?raw=true)

This is an approach to the Facebook Hateful Memes challenge (https://www.drivendata.org/competitions/64/hateful-memes/) for our final Data Science project in ITC

> By Ariela Strimling, Sebastian Kleiner and Rozana Royter

> What is **Israel Tech Challenge**? ITC opens the door for talented professionals from Israel and abroad to develop their careers in technology while focusing on the most in-demand skills in tech. Located in a beautiful campus in Tel Aviv, we offer our students intensive tech training in English, inspired by the IDF’s 8200 unit, and job placement assistance to our graduates. To date, we’ve introduced over 500 alumni to the Israeli hi-tech industry.

## Table of Contents

- [Problem Description](#TheProblem)
- [Overview](#Overview)

## TheProblem:

Our goal is to predict whether a meme is hateful or non-hateful. This is a binary classification problem with **multimodal** input data consisting of the the meme image itself (the image mode) and a string representing the text in the meme image (the text mode).

Why is this task so difficult? As you can see in **Facebook paper** (https://arxiv.org/pdf/2005.04790.pdf), the human is far from being perfect (only AUC of 82.65%!) when it comes to classify images into hateful or not hateful ones. Irony is hard to catch and also sometimes it can be really confusing.

![alt text](https://github.com/SebKleiner/Hateful_Memes/blob/master/fb_scores.JPG?raw=true)

Do you think the following examples are easy for a machine to classify?

![alt text](https://github.com/SebKleiner/Hateful_Memes/blob/master/hateful.JPG?raw=true)

## Overview 

- Facebook provides 10000 images, classified into train set (8500 pictures with a hateful-rate of 35/65), development set (500 images with a ratio of 50/50) and a test set of 1000 images to provide the probability associated to the event.
- We extracted the text, labels and objects detected using **Google Vision API**
- Feature engineering: remove **stop words** (considering both English and Twitter corpuses), **elongated word treatment** (replacing 'niiiiice' with 'nice'), **sentiment analysis** (polarity and subjectivity over the text)
- Data transformation: **CountVectorizer**, **Tfidf vectorizer**, computing **distances** between text/object and text/label to catch irony
- Balancing training dataset
- Modeling: ● Logistic Regression
● Random Forest Classifier
● Naive Bayes Classifier
● AdaBoost Classifier
● Linear Support Vector Classifier
● Logistic regression on top of probabilities obtained ob those classifiers
● Leafs from ensembles as features for a logistic regression
● Simple NN
● Simple RNN's architectures
● RNN: Embedding layer + Bidirectional LSTM

The best model obtained so far is **Logistic regression on top of probabilities obtained ob those classifiers**

## Badges
> Warning: The following badges are for display purposes only and may be considered fake news as they do not reflect actual information about this page. 

[![Fake Coverage](https://camo.githubusercontent.com/3eff610e3559385c77a9b6d87cbe1252cab79a4d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f7665726167652d38302532352d79656c6c6f77677265656e)](https://travis-ci.org/badges/badgerbadgerbadger)  [![A Fake Rating](https://camo.githubusercontent.com/d5cd29c0e2930c3c4026ba87ff427e2e340f461b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f726174696e672d2545322539382538352545322539382538352545322539382538352545322539382538352545322539382538362d627269676874677265656e)](https://travis-ci.org/badges/badgerbadgerbadger)  [![A Fake 3rd Thing](https://camo.githubusercontent.com/b3fc74878a0d5fcca5a78b288aa4b489f65fd7eb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f757074696d652d3130302532352d627269676874677265656e)](https://travis-ci.org/badges/badgerbadgerbadger)
