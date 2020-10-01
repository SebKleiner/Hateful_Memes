import pickle
import pandas as pd
import os


def vectorize(labels, objects, text):
    """
    Reads the tfidf vectorizers from models directory
    Vectorizes labels, objects and text with the correspondant tfidf vectorizer.
    Returns a data frame with the concatenation of the above
    :param labels: list of labesl in a meme
    :param objects: list of objects in a meme
    :param text: text in the meme
    :return: data frame (input for the models)
    """
    with open('models/tfid_labels.pkl', 'rb') as file:
        tfid_labels = pickle.load(file)
    with open('models/tfid_objects.pkl', 'rb') as file:
        tfid_objects = pickle.load(file)
    with open('models/tfid_text.pkl', 'rb') as file:
        tfid_text = pickle.load(file)

    vec_labels = tfid_labels.transform([' '.join(labels)])
    vec_labels = pd.DataFrame(vec_labels.toarray())
    vec_objects = tfid_objects.transform([' '.join(objects)])
    vec_objects = pd.DataFrame(vec_objects.toarray())
    vec_text = tfid_text.transform([text])
    vec_text = pd.DataFrame(vec_text.toarray())

    print(vec_labels)
    print(vec_objects)
    print(vec_text)

    return pd.concat([vec_text, vec_labels, vec_objects], axis=1)


def get_probabilities(input):
    """
    Reads the models from the models directory
    Gets the probabilities of the input for all the models
    :param input: sparse vector
    :return: data frame with the probabilities of the models
    """
    with open('models/ada.pkl', 'rb') as file:
        ada_model = pickle.load(file)
    with open('models/logistic.pkl', 'rb') as file:
        log_model = pickle.load(file)
    with open('models/nai.pkl', 'rb') as file:
        nai_model = pickle.load(file)
    with open('models/rfc.pkl', 'rb') as file:
        rfc_model = pickle.load(file)

    ada_probas = ada_model.predict_proba(input)
    log_probas = log_model.predict_proba(input)
    nai_probas = nai_model.predict_proba(input)
    rfc_probas = rfc_model.predict_proba(input)

    ada_proba = ada_probas.T[1]
    log_proba = log_probas.T[1]
    nai_proba = nai_probas.T[1]
    rfc_proba = rfc_probas.T[1]

    return pd.DataFrame({'rfc': rfc_proba, 'logreg': log_proba, 'nai': nai_proba, 'ada': ada_proba})


def get_prediction(input):
    """
    Reads the model logistic on top model.
    Gets a prediction with the probabilities of the previous models
    :param input: probabilities of previous models
    :return: a prediction 1=hateful, 0=not_hateful
    """
    with open('models/top_logistic.pkl', 'rb') as file:
        top_log_model = pickle.load(file)

    return top_log_model.predict(input)
