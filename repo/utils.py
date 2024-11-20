import openai
import requests
import numpy as np
import time, random

# styles = ["none", "functional", "flowery","informal","formal", "realistic", "magical realism"]
genres = ["none", "passage","joke", "historical fiction", "literary fiction", "science fiction", "mystery", "dystopian", "horror","story"]
subjects = ["none", "love", "strawberry", "programmer"]
moods = ["none", "happy", "sad", "boring","moved"]
plots = ["none", "Overcoming the Monster", "Voyage and Return"]

path = "../data/cases/pool.txt"
new_path = "../data/cases/diy_pool.txt"
with open("key.txt", 'r', encoding='utf-8') as f:
    keys = [i.strip() for i in f.readlines()]

def list2text(item_list,type='word'):
    if type == 'phrase':
        temp = ""
        for i in range(len(item_list)):
            temp += str(i+1) + ". " + item_list[i] + "\n"
        return temp.strip()
    elif len(item_list) == 1:
        return item_list[0]
    elif len(item_list) == 2:
        return ' and '.join(item_list)
    elif len(item_list) > 2:
        return','.join(item_list[:-1]) + " and "+item_list[-1]

    
def diy():
    index_num = random.randint(0, len(keys)-1)
    data = []
    while True:
        question = input()
        data.append([question, None])
        response = requests.post("xxx", json={
            "data": [
                data,
                keys[index_num],
            ]
        }).json()

        resp = response["data"][0]
        print(resp[-1][1])
        data[-1][1] = resp[-1][1]

def load_pool(path):
    pool = open(path, encoding='utf8').readlines()
    pool = [line.strip().split("\t") for line in pool]
    return pool

def make_prompt(examples=[], conditions={}):
    def merge_list(item_list):
        if len(item_list)<=2:
            merged = " and ".join(item_list)
        else:
            merged = ", ".join(item_list[:-1]) + " and " + item_list[-1]
        return merged


    genre, subject, mood, plot = conditions['genre'], conditions['subjects'],  conditions['mood'], conditions['plots']
    # style, genre, subject, mood, plot = conditions['style'], conditions['genre'], conditions['subjects'],  conditions['mood'], conditions['plots']
    genre = merge_list(genre)
    plot = merge_list(plot)
    subject = merge_list(subject)

    if len(examples) == 0:
        if mood == "none":
            if plot == "none":
                prompt = "Write a " + genre + " about " + subject +".\n"
                # prompt = "Write a " + style + " " + genre + " about " + subject +".\n"
            else:
                prompt = "Write a " + genre + " about " + subject + ", with a \"" + plot+"\" plot.\n"
                # prompt = "Write a " + style + " " + genre + " about " + subject + ", with a \"" + plot+"\" plot.\n"
        else:
            if plot == "none":
                prompt = "Write a " + genre + " about " + subject +" that makes the readers feel " + mood + ".\n"
                # prompt = "Write a " + style + " " + genre + " about " + subject +" that makes the readers feel " + mood + ".\n"
            else:
                prompt = "Write a " + genre + " that makes the readers feel " + mood + ". It describes the following subjects: "+ subject + " . It should contain the following plots: " + plot+"."
                # prompt = "Write a " + style + " " + genre + " that makes the readers feel " + mood + ". It describes the following subjects: "+ subject + " . It should contain the following plots: " + plot+"."

    elif len(examples) == 1:
        example = examples[0]
        example['genre'] = merge_list(example['genre'])
        if mood == "none":
            if plot == "none":
                    prompt = "Here is an example of writing a " + example['genre']+ " about " + example[
                    # prompt = "Here is an example of writing a " + example['style']+ " " + example['genre']+ " about " + example[
                'mood'] + ": " +example['plot_summary']+ "\n Draw inspiration from the given example, please write a " + genre + " about " + subject + ".\n"
            else:
                prompt = "Here is an example of writing a " + example['genre']+ " about " + example['subjects']+ ": " + example[
                             'plot_summary'] + "\n Draw inspiration from the given example, please write a " + genre + " about " + subject + ", with a \"" + plot+"\" plot.\n"
        else:
            if plot == "none":
                prompt = "Here is an example of writing a " + example['genre']+ " that makes the readers feel " + example[
                         'mood'] +". It describes the following subjects: " + example[
                'subjects'] + ". Its main plots are:"+example['plot_summary']+ "\n Draw inspiration from the given example, please write a " + genre + " about " + subject + " that makes the readers feel " + mood + ".\n"
            else:
                prompt = "Here is an example of writing a " + example['genre'] + " that makes the readers feel " + example[
                         'mood'] + ". It describes the following subjects: " + example[
                'subjects'] +". Its story is: "+example['plot_summary'] +  "\n Draw inspiration from the given example, please write a " + genre + " that makes the readers feel " + mood + ". It describes the following subjects: "+ subject + " . It should at least contain the following plots (the more interesting plots the better): " + plot+"."

    elif len(examples) > 1:
        few_shots = ""
        if mood == "none":
            if plot == "none":
                for example in examples:
                    example['genre'] = merge_list(example['genre'])
                    few_shots += "Here is an example of writing a " + example['genre']+ " about " + example[
                        'subjects'] + ": " + example['plot_summary']+ "\n"
                prompt = few_shots + "Draw inspiration from the given examples, please write a " + genre + " about " + subject + ".\n"
            else:
                for example in examples:
                    example['genre'] = merge_list(example['genre'])
                    few_shots += "Here is an example of writing a " + example['genre']+ " about " + example[
                        'subjects'] + ", with a \"" + plot+"\" plot: " + example[4]+"\n"
                prompt = few_shots + "Draw inspiration from the given examples, please write a " + genre + " about " + subject + ", with a \"" + plot+"\" plot.\n"
        else:
            if plot == "none":
                for example in examples:
                    example['genre'] = merge_list(example['genre'])
                    few_shots += "Here is an example of writing a " + example['genre']+ " about " + example[
                        'subjects'] + " that makes the readers feel " + example['mood']+ ": " + example['plot_summary']+ "\n"
                prompt = few_shots + "Draw inspiration from the given examples, combining their plots, please write a " + genre + " about " + subject + " that makes the readers feel " + mood + ".\n"
            else:
                for example in examples:
                    example['genre'] = merge_list(example['genre'])
                    few_shots += "Here is an example of writing a " + example['genre'] + " that makes the readers feel " + example[
                         'mood'] + ". It describes the following subjects: " +example[
                        'subjects'] + ". Its storyline is: " +example['plot_summary'] + "\n"
                prompt = few_shots + "Draw inspiration from the given examples, please write a " + genre + " that makes the readers feel " + mood + ". It describes the following subjects: "+ subject + " . It should at least contain the following plots (the more interesting plots the better): " + plot+"."


    return prompt.replace("  ", " ").replace(" <br>",". ").replace("<br>","").replace("..",".")


def generate(prompt):
    index_num = random.randint(0, len(keys)-1)
    openai.api_key = keys[index_num] 

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    generation = response['choices'][0]['message']['content']

    return generation


def build_pool(path, new_path):
    current_pool = load_pool(path)
    new_pool = []
    #for style in styles:
    for genre in genres:
        for mood in moods:
            for subject in subjects:
                for example in current_pool:
                    try:
                        prompt = make_prompt([example], [genre, subject, mood])#
                        # prompt = make_prompt([example], [style, genre, subject, mood])
                        print("Prompt: " + prompt)
                        generation = generate(prompt)
                        print("Generation: " + generation)
                        new_pool.append([generation, genre, subject, mood])
                        # new_pool.append([generation, style, genre, subject, mood])
                    except:
                        pass
    pool = current_pool + new_pool
    with open(new_path, "w", encoding="utf8") as f:
        for item in pool:
            f.write("\t".join(item) + "\n")


def retrieve_example(pool, conditions, num_results=1):
    def find_most_similar_items(items, new_item, num_results=num_results):
        if num_results == 0:
            return []
        similarities = []
        for item in items:
            # Convert tuples to numpy arrays for easier computation
            item_vec = np.array(item)
            new_item_vec = np.array(new_item)
            # Compute cosine similarity between item and new_item
            similarity = np.dot(item_vec, new_item_vec) / (np.linalg.norm(item_vec) * np.linalg.norm(new_item_vec))
            similarities.append(similarity)
        # Get indices of top num_results items with highest similarity
        top_indices = np.argsort(similarities)[::-1][:num_results]
        # Return top items
        return top_indices

    def create_vector(item):
        vector = []
        for i in range(4):
            #if i == 0:
            #    global_list = styles
            if i == 0:
                global_list = genres
            elif i == 1:
                global_list = subjects
            elif i == 2:
                global_list = moods
            elif i == 3:
                global_list = plots
            try:
                vector.append(global_list.index(item[i]))
            except:
                vector.append(global_list.index("none"))
        return vector

    pool_indexs = []
    for item in pool:
        pool_indexs.append(create_vector(item))

    new_item_index = create_vector(conditions)
    top_indices = find_most_similar_items(pool_indexs, new_item_index)
    examples = [pool[indice] for indice in top_indices]

    return examples


def main(path):
    pool = load_pool(path)
    queries = {'mood':'excited, thrilled','genre':['action'],'subjects':['adventure','death'],'plots':['he escaped from the island']}
    # queries = {'mood':'excited, thrilled','style':'fiction,action','genre':['action'],'subjects':['adventure','death'],'plots':['he escaped from the island']}

    conditions = ["realistic", "story","society","shocked","none"]  # style, genre, subject, tone, mood
    #conditions = ["romantic", "story","gay","authentic","moved","Peter discovers Kate’s secret of being gay in a letter and later supports her through her illness until her ultimate decision to end her suffering."]  # style, genre, subject, tone, mood
    #conditions = ["informal", "joke", "love", "humorous", "happy"]  # style, genre, subject, tone, mood
    best_examples = retrieve_example(pool, conditions,num_results=1)
    prompt = make_prompt(examples=best_examples,conditions=queries)
    print("Prompt: " + prompt)
    generation = generate(prompt)
    print("Generation: " + generation)

#diy()
#main(new_path)
#build_pool(path, new_path)
