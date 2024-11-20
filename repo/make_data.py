import re
import time
import json
import requests
import pandas as pd
import random
from random import randint


def ask_chatgpt(prompt,question,key):
    data = []
    answers = []
        #print("user: ", end="")
    question = prompt + question
    data.append([question, None])

    #print("assistant: ", end="")
    response = requests.post("xxx", json={
        "data": [
            data,
            key,
        ]
    }).json()
    resp = response["data"][0]
    answers.append(resp[-1][1])
    print(resp[-1][1])
    data[-1][1] = resp[-1][1]

    return answers

def fetch_info():
    with open("key.txt", 'r', encoding='utf-8') as f:
        keys = [i.strip() for i in f.readlines()]
    df_details = pd.read_json('../data/IMDB_movie_details.json', lines=True)

    movie_data = []

    for i in range(df_details.shape[0]):
        item = {}
        item['genre'] = df_details['genre'][i]
        item['plot_synopsis'] = re.sub('Written by.*','',df_details['plot_synopsis'][i]).strip()
        item['plot_summary'] = df_details['plot_summary'][i]

        prompt1 = "Here is a story:\n" + item['plot_summary'] + "\n Answer the following question based on the above story: \n"
        # question1 = 'use one or two words to answer the writing style of this story'
        #try:
        #    answers = ask_chatgpt(prompt1,question1,keys[0])
        #    item['style'] = answers[0]
        #except:
        #    item['style'] = None

        question2 = 'use one or two words to answer the moods that the readers may feel'
        try:
            answers = ask_chatgpt(prompt1, question2,keys[1])
            item['mood'] = answers[0]
        except:
            item['mood'] = None
        #time.sleep(21)
        question3 = 'give a list of distinctive subjects this story is trying to portray'
        try:
            answers = ask_chatgpt(prompt1, question3,keys[2])
            item['subjects'] = answers[0]
        except:
            item['subjects'] = None

        prompt2 = "Here is a story:\n" + item['plot_synopsis'] + "\n Answer the following question based on the above story: \n"
        question4 = 'summarize the above story and give an organized list of sentences, each of which describes one plot'
        try:
            answers = ask_chatgpt(prompt2, question4,keys[3])
            item['plots'] = answers[0]
            #item['plots'] = None
        except:
            item['plots'] = None
        time.sleep(5)
        movie_data.append(item)
        #break
    print(movie_data)


def continue_fetch_info():
    with open("key.txt", 'r', encoding='utf-8') as f:
        keys = [i.strip() for i in f.readlines()]
    with open('../data/movie_data.json', 'r') as f:
        movie_data = json.load(f)

    topics = ['mood','subjects','plots']
    #topics = ['style','mood','subjects','plots']
    questions = [
    # questions = ['use one or two words to answer the writing style of this story',
    'use one or two words to answer the moods that the readers may feel about the story',
    'give a list of distinctive subjects this story is trying to portray',
    'summarize the above story and give an organized list of sentences, each of which describes one plot'
    ]

    while True:
        done = True
        new_movie_data = []
        for index,item in enumerate(movie_data):
            print(index)
            for key_index,(topic,question) in enumerate(zip(topics,questions)):
                if item[topic] == None:
                    done = False
                    if topic == 'plots':
                        prompt = "Here is a story:\n" + item['plot_summary'] + "\n Answer the following question based on the above story: \n"
                    else:
                        prompt = "Here is a story:\n" + item['plot_summary'] + "\n Answer the following question based on the above story: \n"
                    try:
                        index_num = randint(0, 8)
                        answer = ask_chatgpt(prompt, question,keys[index_num])
                        item[topic] = answer[0]
                        #time.sleep(5)
                    except:
                       continue

            new_movie_data.append(item)
        movie_data = new_movie_data
        with open('../data/new_movie_data.json', 'w') as fout:
            json.dump(movie_data, fout)
        if done:
            break
    print('Done')

    with open('../data/new_movie_data.json', 'w') as fout:
        json.dump(movie_data, fout)

def fix():
    with open("key.txt", 'r', encoding='utf-8') as f:
        keys = [i.strip() for i in f.readlines()]
    with open('../data/movie_data.json', 'r') as f:
        movie_data = json.load(f)

    topics = ['mood','subjects','plots']

    questions = [
        # questions = ['use one or two words to answer the writing style of this story',
        'use one or two words to answer the moods that the readers may feel about the story',
        'give a list of distinctive subjects this story is trying to portray',
        'summarize the above story and give an organized list of sentences, each of which describes one plot'
    ]

    while True:
        done = True
        new_movie_data = []
        for index,item in enumerate(movie_data):
            print(index)
            if item['plot_synopsis'] == "":
                done = False
                prompt = "Here is a story:\n" + item['plot_summary'] + "\n Answer the following question based on the above story: \n"
                try:
                    index_num = randint(0, 8)
                    answer = ask_chatgpt(prompt, 'summarize the above story and give an organized list of sentences, each of which describes one plot',keys[index_num])
                    item['plots'] = answer[0]
                    item['plot_synopsis'] == "replaced by plot_summary"
                    #time.sleep(5)
                except:
                   continue

            new_movie_data.append(item)
        movie_data = new_movie_data
        with open('../data/new_new_movie_data.json', 'w') as fout:
            json.dump(movie_data, fout)
        if done:
            break
    print('Done')

    with open('../data/new_movie_data.json', 'w') as fout:
        json.dump(movie_data, fout)

#fix()

#continue_fetch_info()