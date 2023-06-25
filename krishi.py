import numpy as np
import pickle
import json
import timeit
import random
import pyttsx3
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

print("Hey user I am KrishiBot How can I help you!!\n")
engine.say("Hey user I am Krishibot How can I help you")
engine.runAndWait()

def cos_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    cosine = cosine_similarity(v1, v2)
    # print(cosine)
    return cosine

def generateResponse(jsonDataFile, minScore, cosine, max):

    responseIndex = 0
    if (max > minScore):
        new_max = max - 0.01
        list = np.where(cosine > new_max)

        responseIndex = random.choice(list[0])
    else:
        return "Chat with Krishibot...", 0

    j = 0

    with open(jsonDataFile, "r") as sentences_file:
        reader = json.load(sentences_file)
        for row in reader:
            j += 1  # we begin with 1 not 0 & j is initialized by 0
            if j == responseIndex:
                return row["answer"], max
                break


def krishibot(testSetDocument, minScore, jsonDataFile, piklePathTfidfVectorizer, piklePathTfidfMatrixTrain):
    testSet = (testSetDocument, "")

    try:

        f = open(piklePathTfidfVectorizer, 'rb')  # to use
        tfidfVectorizer = pickle.load(f)
        f.close()

        f = open(piklePathTfidfMatrixTrain, 'rb')
        tfidfMatrixTrain = pickle.load(f)
        f.close()

    except:

        tfidfVectorizer, tfidfMatrixTrain = trainKrishibot(jsonDataFile, piklePathTfidfVectorizer,
                                                           piklePathTfidfMatrixTrain)  # to train

    tfidfMatrixTest = tfidfVectorizer.transform(testSet)
    # print(tfidfMatrixTest)

    # print("\n",tfidfMatrixTrain)
    cosine = cos_similarity(tfidfMatrixTest, tfidfMatrixTrain)
    # print("\n",cosine)
    cosine = np.delete(cosine, 0)
    max = cosine.max()

    answer, score = generateResponse(jsonDataFile, minScore, cosine, max)
    return answer, score


def dataFromJsonFile(jsonDataFile, document):
    i = 0
    with open(jsonDataFile, "r") as sentences_file:
        reader = json.load(sentences_file)
        # reader.next()
        # reader.next()
        for row in reader:
            # if i==stop_at_sentence:
            #    break
            document.append(row["question"])
            i += 1
    return document


def trainKrishibot(jsonDataFile, tfidfVectorizerPikleFile, tfidfMatrixTrainPikleFile):
    document = []

    document.append(" Hi Krishibot.")  # enter your test sentence
    document.append(" Hi Krishibot.")

    start = timeit.default_timer()


    document = dataFromJsonFile(jsonDataFile, document)

    tfidfVectorizer = TfidfVectorizer()
    # print(tfidfVectorizer)
    tfidfMatrixTrain = tfidfVectorizer.fit_transform(document)  # finds the tfidf score with normalization

    stop = timeit.default_timer()
    # print ("training time took was : ")
    # print (stop - start)

    f = open(tfidfVectorizerPikleFile, 'wb')
    pickle.dump(tfidfVectorizer, f)
    f.close()

    f = open(tfidfMatrixTrainPikleFile, 'wb')
    pickle.dump(tfidfMatrixTrain, f)
    f.close()

    return tfidfVectorizer, tfidfMatrixTrain


def krishiChats(query):
    dirpath = os.path.dirname(os.path.realpath(__file__))
    jsonDataFile = dirpath + "/agricultureFAQ.json"
    tfidfVectorizerPikleFile = dirpath + "/previous_tfidf_vectorizer.pickle"
    tfidfMatrixTrainPikleFile = dirpath + "/previous_tfidf_matrix_train.pickle"
    minScore = 0.6
    answer, score = krishibot(query, minScore, jsonDataFile, tfidfVectorizerPikleFile, tfidfMatrixTrainPikleFile)
    return answer

def ambiguityResolve(query):
    ambiguousQuery = []
    dirpath = os.path.dirname(os.path.realpath(__file__))
    jsonDataFile = dirpath + "/agricultureFAQ.json"
    with open(jsonDataFile, "r") as file:
        reader = json.load(file)
        for ambiguous in reader:

            question = str(ambiguous["question"])
            if question.__contains__(query):
                ambiguousQuery.append(ambiguous["question"])

    return ambiguousQuery

while True:
    query = input("Krishibot : ")
    krishiBotReply = krishiChats(query)
    if(krishiBotReply=='Chat with Krishibot...'):
        ambiguousQuery = ambiguityResolve(query)
        for i in ambiguousQuery:
            print(i)
            engine.say(i)
            engine.runAndWait()


    else:
        print(krishiBotReply+"\n")
        engine.say(krishiBotReply)
        engine.runAndWait()
