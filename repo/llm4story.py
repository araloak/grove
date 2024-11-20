import pickle
import json
import random
import os, re
import torch
import time
import sys
import os
import traceback
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from utils import *

with open("key.txt", 'r', encoding='utf-8') as f:
    keys = [i.strip() for i in f.readlines()]

def get_best_example(all_recommendations,n_best=1):
    final_scores = []
    for id in range(len(all_recommendations['plot_recommend'])):
        overall_score = 0
        for condition in all_recommendations:
            try:
                overall_score += all_recommendations[condition][id][-1]
            except Exception as e:
                pass
        final_scores.append((id,overall_score))
    best_ids = [i[0] for i in sorted(final_scores, key=lambda x: x[1], reverse=True)]
    if n_best == 1:
        return best_ids[0]
    else:
        return best_ids[0:n_best]
def multi_corpus_recommend(entry_name,query,corpus,k=4):
    if not os.path.exists('../data/'+entry_name+'.pkl'):
        first = True
        corpus_embeddings = {}
    else:
        first = False
        with open('../data/'+entry_name+'.pkl', 'rb') as f:
            corpus_embeddings = pickle.load(f)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='../models/sbert')
    query_embedding = model.encode(query)
    hits = []
    for i, item in enumerate(corpus):
        if item == []:
            hits.append({'corpus_id': i, 'score': 0})
            continue

        if first:
            item_embeddings = model.encode(item)
            corpus_embeddings[str(i)] = item_embeddings
        else:
            item_embeddings = corpus_embeddings[str(i)]

        similarity_sum = 0
        for sentence_embedding in item_embeddings:
            similarity = util.cos_sim(sentence_embedding, query_embedding).mean(-1)
            similarity_sum += similarity
        similarity_sum/=len(item)
        hits.append({'corpus_id': i, 'score': similarity_sum})
    if first:
        file = open('../data/'+entry_name+'.pkl', 'wb')
        pickle.dump(corpus_embeddings, file)
        file.close()

    #hits = sorted(hits, key=lambda x: x['score'], reverse=True)#[:k]
    scores = np.array([i['score'].item() for i in hits]).reshape(-1, 1)
    scaler = MinMaxScaler()  
    scaler = scaler.fit(scores)  
    scalered_scores = scaler.transform(scores)  
    recommendation = []
    # recommendation = [(i['corpus_id'],corpus[i['corpus_id']], scalered_scores[index][0]) for index,i in enumerate(hits)]
    for index, i in enumerate(hits):
        try:
            recommendation.append((i['corpus_id'], corpus[i['corpus_id']], scalered_scores[index][0]))
        except Exception as e:
            pass
    return recommendation


def single_corpus_recommend(entry_name,query,corpus,k=4):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='../models/sbert')
    if not os.path.exists('../data/'+entry_name+'.bin'):
        corpus_embeddings = model.encode(corpus)
        torch.save(corpus_embeddings,'../data/'+entry_name+'.bin')
    else:
        corpus_embeddings = torch.load('../data/'+entry_name+'.bin')
    query_embeddings = model.encode(query)

    # Find the top-2 corpus documents matching each query
    hits = util.semantic_search(query_embeddings, corpus_embeddings,top_k=len(corpus))
    #print(hits)
    scores = np.array([i['score'] for i in hits[0]]).reshape(-1, 1)
    scaler = MinMaxScaler()  
    scaler = scaler.fit(scores)  
    scalered_scores = scaler.transform(scores)  
    recommendation = []
    for index,i in enumerate(hits[0]):
        try:
            recommendation.append((i['corpus_id'], corpus[i['corpus_id']], scalered_scores[index][0]) )
        except Exception as e:
            pass

    # recommendation = [(i['corpus_id'], corpus[i['corpus_id']], scalered_scores[index][0]) for index,i in enumerate(hits[0])]

    return recommendation

def clean_split(text):
    sentences = []
    if text is None:
        return sentences
    text = text.split('\n')
    for line in text:
        line = re.sub('\n','',line)
        line = re.sub('[1-9]*\.','',line)
        line = re.sub('\(*[1-9]*\)','',line)
        line = line.replace("- ","")
        line = line.replace("  "," ").strip()
        if line!="":
            sentences.append(line)
    if 'an AI language model' in sentences[0]:
        sentences = sentences[1:]
    return sentences

def get_all_recommend(movie_data, queries,topk=4):
    all_recommendations = {}

    all_moods, all_subjects, all_genres, all_plots = [], [], [], []
    # all_styles, all_moods, all_subjects, all_genres, all_plots = [], [], [], [], []
    for i in range(len(movie_data)):
        mood = 'None' if movie_data[i]['mood'] is None else movie_data[i]['mood']
        # style = 'None' if movie_data[i]['style'] is None else movie_data[i]['style']
        # all_styles.append(style)
        all_moods.append(mood)

        all_genres.append(movie_data[i]['genre'])

        # if movie_data[i]['plot_summary'] == '':
        #     plots = movie_data[i]['plots']
        # else:
        #     plots = clean_split(movie_data[i]['plots'])

        plots = clean_split(movie_data[i]['plots'])
        subjects = clean_split(movie_data[i]['subjects'])

        all_subjects.append(subjects)
        all_plots.append(plots)

    all_recommendations['mood_recommend'] = single_corpus_recommend('mood',queries['mood'], all_moods, k=topk)
    #all_recommendations['style_recommend'] = single_corpus_recommend('style',queries['style'], all_styles, k=topk)

    all_recommendations['genre_recommend'] = multi_corpus_recommend('genre',queries['genre'], all_genres, k=topk)
    all_recommendations['subject_recommend'] = multi_corpus_recommend('subjects',queries['subjects'], all_subjects, k=topk)
    all_recommendations['plot_recommend'] = multi_corpus_recommend('plots',queries['plots'], all_plots, k=topk)

    return all_recommendations

def ask_why(story,new_info=[None],plots=None,depth=0,width=1,unclarities=None, queries = {}):

    def summarize(story):
        prompt = 'Here is a story: "' + story +'"\n' + 'When analyzing factional stories, it is okay to mention negative aspects. Pretend to be a writer, without further ado, point out '+str(width)+' missing background information in the story with '+str(width)+' simple sentence.\n'
        story_summary = generate(prompt)
        return story_summary

    if plots == None:
        unclarities = summarize(story)
        plots = [unclarities]
        plots = clean_split(unclarities)
        new_info.append([(story,unclarities)])
        #new_info.append([(story,plots[1])])

    new_plots = []
    for index,unclarity in enumerate(plots):
        prompt_1 = 'Here is a story: \n' + story + '\n' + 'A missing detail is: \n' + unclarity + '\n' + \
                   'Except for pure coincidence, point out ' + str(1) + \
                   ' factual pieces of background information that compensate the story one by one. Each additional piece of information should be in one short sentence and only contains factual information without opinions or judgments.'

        #prompt_1 = 'Here is a story: \n' + story +'\n' + 'An unclarity is: \n'+ unclarity +'\n' + 'Except for pure coincidence and subject reasons, reveal me some implicit background knowledge within one or two sentencese to rationalize the story. The additional information should be short and imply the topic: '+ list2text(queries['subjects']) + "."
        explanations = [generate(prompt_1)]
        #explanations = clean_split(explanations)
        #prompt_2 = 'Here is a story: \n' + story +'\n' + 'Here is a plot: \n'+plot + '\n'+ 'Here are a list of possible reasons about why the above plot is reasonable in the story: \n' + list2text(explanations,type='phrase') + '\n' + ' pretend to be a professional writer, please select the reason that is the closest to the '+ list2text(queries['subjects']) + ' and only output the index number without any explanation.'
        #explanation_num = random.randint(0, len(explanations)-1)
        #output = generate(prompt_2)
        #explanation_num = int(re.sub("[/.,a-zA-Z]*","",output).strip()) - 1
        new_plots+=explanations # pick an explanation
        for index,explanation in enumerate(explanations):
            current_index = len(new_info) + 1
            if current_index % width == 0:
                new_info.append(new_info[int(current_index / width)-1] + [(unclarity,explanation)])
            elif current_index % width !=0:
                new_info.append(new_info[int(float((current_index-1) / width))-1] + [(unclarity, explanation)])
            else:
                new_info.append([(unclarity, explanation)])

    if depth < 1:
        depth += 1
        unclarities,new_info = ask_why(story, new_info, new_plots, depth,width, unclarities)
        return unclarities,new_info
    return unclarities,new_info

def add_new_info(story,picked_info):
    num_parts = 2
    new_story = []
    # reasons = [each[1] for each in new_info]
    # reasons = "\n".join(reasons)
    prompt = 'Here is a story: \n' + story + '\n' + 'Here is the missing background information: \n' + picked_info + '\n' + \
             'Pretend to be a writer and complete the story by including the given information. Modify the necessary sentences in the story and repeat the unrelated parts to include the given background information.'

    messages = [{"role": "user", "content": prompt}]
    for part in range(num_parts):
        index_num = random.randint(0, len(keys) - 1)
        openai.api_key = keys[index_num]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        generation = response['choices'][0]['message']['content']

        new_story.append(generation)
        question = 'Without further ado, start generating the second half of the story.'
        messages.append({"role": "assistant", "content": generation})
        messages.append({"role": "user", "content": question})

    new_story = "\n".join(new_story)
    return new_story
def pick_info(story,new_info,queries = {}):
    chain_of_reasons = []
    for index in range(3,7):
        chain_of_reason = []
        for each in list(reversed(new_info[index])):
            chain_of_reason.append(each[-1])
        #chain_of_reason.append(each[0])
        chain_of_reason = list(reversed(chain_of_reason))
        chain_of_reasons.append(". ".join(chain_of_reason).replace("..","."))

    prompt = 'Here is a story: \n' + story + '\n' + 'Here are a list of reasons, explaining an unclarity in the story in depth: \n' + list2text(chain_of_reasons,type='phrase') + '\n' + 'Pretend to be a professional writer, please select the reason that is the closest to the '+ \
             list2text(queries['subjects']) + ' and only output the index number without any explanation.'
    reason_id = int(re.sub("[/.,a-zA-Z]*","",generate(prompt)).strip()) - 1
    return chain_of_reasons[reason_id]

    #return picked_info
plot_kind = []
with open("inputs/plot.txt", 'r', encoding='utf-8') as file:
    reddit_plot = [k.strip() for k in file.readlines()]
    # print(len(reddit_plot))
    for i in range(len(reddit_plot)):
        j = reddit_plot[i].find('[ ')
        if j != -1:# kind:[_IP/WP/FF/EU/CW/RF/OT/PI/Wp/PM_]_
            plot_kind.append(reddit_plot[i][j+2:j+4])
            reddit_plot[i] = reddit_plot[i][0:j].strip() + reddit_plot[i][j+6:].strip()
        else:
            plot_kind.append('')

suject = ['lovers', 'cats', 'survivors']
genre = ['Historical Fiction', 'Literary Fiction', 'Science Fiction']
mood = ['happy','angry','fearful','sad']
num_kind = len(suject) * len(mood) * len(genre)
# mood = ['angry', 'fearful', 'happy', 'sad']

def generate_result(simple_plot, output_file, s):
    request = []
    for count in range(len(suject) * len(mood) * len(genre)):
        request.append({'plots': [simple_plot]})
    count = 0
    for a in mood:# 'mood': [], 'genre': [], 'subjects': [],
        for b in genre:
            for c in suject:
                request[count]['mood'] = a
                request[count]['genre'] = [b]
                request[count]['subjects'] = [c]
                count += 1
    # queries = {'mood':'excited, thrilled','style':'magical','genre':['action'],'subjects':['adventure','death']}
    # queries['plots'] = [simple_plot]

    for queries in request[s%num_kind:]:  #
        flag = 0
        start = time.time()
        for i in range(20):  
            if flag == 1:
                # save
                output_file.write("plot:" + simple_plot)
                output_file.write("\n")
                output_file.write(prompt_before_search)
                output_file.write("\n\n-----------------------------------------------\n\n")
                output_file.write(story_before_search)
                output_file.write("\n\n-----------------------------------------------\n\n")
                # output_file.write(prompt)
                # output_file.write("\n\n-----------------------------------------------\n\n")
                # output_file.write(story)
                # output_file.write("\n\n-----------------------------------------------\n\n")
                output_file.write(str(new__info))
                output_file.write("\n\n-----------------------------------------------\n\n")
                output_file.write(str(picked_info))
                output_file.write("\n\n-----------------------------------------------\n\n")
                output_file.write(new_story)
                output_file.write("\n\n-----------------------------------------------\n\n")
                output_file.write("********************************************************************\n")
                s += 1

                break
            else:
                try:
                    # print("plot:"+simple_plot)

                    # queries = {'mood':'excited, thrilled','style':'magical','genre':['action'],'subjects':['adventure','death'],
                    # 'plots':['he escaped from the island','he met a mermaid','he had a soup']}

                    # all_recommendations = get_all_recommend(movie_data, queries, 4)

                    prompt_before_search = make_prompt(conditions=queries)  
                    # print(prompt_before_search)
                    # print("\n\n-----------------------------------------------\n\n")

                    story_before_search = generate(prompt_before_search).replace("\n\n", "\n")
                    # print(story_before_search)
                    # print("\n\n-----------------------------------------------\n\n")

                    # example_id = get_best_example(all_recommendations)

                    # example_id_1 = all_recommendations['plot_recommend'][0][0]

                    # example = movie_data[example_id]
                    # examples = [example]
                    # examples = []
                    # examples = [example_1, example_2]

                    # prompt = make_prompt(examples=examples, conditions=queries)
                    # print(prompt)
                    # print("\n\n-----------------------------------------------\n\n")
                    # story = generate(prompt).replace("\n\n", "\n")
                    # print(story)
                    # print("\n\n-----------------------------------------------\n\n")

                    story_summary, new__info = ask_why(story_before_search, new_info = [None], queries = queries)  # , plot = simple_plot
                    # print(new__info)

                    # print("\n\n-----------------------------------------------\n\n")
                    # for key in new_info:
                    # new_info = new_info[-4:]
                    # print(Exception)
                    picked_info = pick_info(story_before_search, new__info, queries = queries)
                    # print(picked_info)
                    # print("\n\n-----------------------------------------------\n\n")
                    # new_info = []
                    new_story = add_new_info(story_before_search, picked_info).replace("\n\n", "\n")

                    print("plot:" + simple_plot + '\n用时：' + "{:.2f} s".format(time.time() - start))
                    # print(new_info)
                    flag = 1
                except Exception as e:
                # except IOError as e:
                    # print(openai.api_key)
                    print(str(e))

                    if 'list index out of range' in str(e):
                        print(traceback.format_exc())
                        # print(keys[0][0])
                        # print(str(openai.api_key))
                    elif 'You exceeded your current quota' in str(e):
                        delete = open("exceeded_keys.txt", 'a+', encoding='utf-8')
                        delete.write(openai.api_key + '\n')
                        delete.close()
                        if str(openai.api_key) in keys:
                            keys.remove(str(openai.api_key))
                    elif 'Limit: 3 / min. Please try again in 20s'  in str(e) or \
                            'That model is currently overloaded with other requests' in str(e):
                        time.sleep(40)  
                    elif 'The server had an error while processing your request. Sorry about that! You can retry your request'  in str(e):
                        os.system("python llm4story.py")
                    flag = 0
                    if time.time() - start > 280.0:  
                        break
        if time.time() - start > 280.0:  
            time.sleep(60)
            break
    return s

def write_file(num):
    with open("../data/ablation/_retrieval/res" + str(num + 1) + "/results.txt", 'r', encoding='utf-8') as output_file:
        content = output_file.read()
    content = content.split('********************************************************************\n')
    #content = content[:-1]
    # num_queries = len(content)
    file_list = [
                'prompt_before_search.txt',
                'story_before_search.txt',
                'prompt.txt',
                'story.txt',
                'new_info.txt',
                'picked_info.txt',
                'new_story.txt'
                ]
    for i in range(len(file_list)):  
        with open("../data/ablation/_retrieval/res" + str(num + 1) + "/" + file_list[i], 'w', encoding='utf-8') as out:
            for each in content: 
                each_plot = [x.replace('\n', '') for x in
                             each.split('-----------------------------------------------')]
                if i == 0:  
                    # print(each_plot)
                    project = each_plot[i].strip()
                    sub = project.find('Write a ')
                    project = project[sub:]
                    out.write(project + '\n')
                    # print(project)
                else:
                    try:
                        project = each_plot[i].strip() 
                        out.write(project + '\n')
                    except:
                        pass
def extract_queries(prompt):
    queries = {}

    remaining, plot = prompt.split("(the more interesting plots the better):")
    queries["plots"] = [plot.strip()]

    for sub in suject:
        if sub in remaining:
            queries['subjects'] = [sub]

    for mo in mood:
        if mo in remaining:
            queries['mood'] = mo


    for ge in genre:
        if ge in remaining:
            queries['genre'] = [ge]

    return queries

if __name__ == '__main__':
    with open('../data/movie_data.json', 'r') as fi:
        movie_data = json.load(fi)
    ablation = 'case'
    with open("../data/ablation/" + ablation + "/prompt.txt", "r", encoding="utf-8") as f:
        prompts = [i.strip() for i in f.readlines()]

    try:
        with open("../data/ablation/" + ablation + "/new_stories.json",'r') as file:
            new_stories = json.load(file)
        with open("../data/ablation/" + ablation + "/direct_gen_stories.json",'r') as file:
            direct_gen_stories = json.load(file)
        # with open("../data/ablation/" + ablation + "/w_retrieve_stories.json",'r') as file:
        #     w_retrieve_stories = json.load(file)
        with open("../data/ablation/" + ablation + "/picked_info.json",'r') as file:
            picked_infos = json.load(file)


        count = len(new_stories)
    except:
        new_stories = {}
        direct_gen_stories = {}
        picked_infos = {}
        w_retrieve_stories = {}
        count = 0
    while count<100:
        for index,prompt in enumerate(prompts):
            #print(index)
            # queries = {'mood': 'excited, thrilled', 'style': 'magical', 'genre': ['action'],
            #            'subjects': ['adventure', 'death'],
            if str(index) in new_stories.keys():
                count+=1
                print(index,"done")
                continue
            try:
                queries = extract_queries(prompt)
                all_recommendations = get_all_recommend(movie_data, queries, 4)

                #prompt_before_search = make_prompt(conditions=queries)  
                # print(prompt_before_search)
                # print("\n-----------------------------------------------\n")

                #story_before_search = generate(prompt).replace("\n", "\n")
                # print(story_before_search)
                # print("\n-----------------------------------------------\n")

                example_id = get_best_example(all_recommendations)

                # example_id_1 = all_recommendations['plot_recommend'][0][0]

                example = movie_data[example_id]
                examples = [example]
                # examples = []
                # examples = [example_1, example_2]

                prompt = make_prompt(examples=examples, conditions=queries)
                #prompt = "Pretend to be a writer, " + prompt
                # print(prompt)
                # print("\n-----------------------------------------------\n")
                story = generate(prompt).replace("\n", " ").replace("  "," ")
                # print(story)
                # print("\n-----------------------------------------------\n")

                story_summary, new__info = ask_why(story, new_info=[None], queries=queries)  # , plot = simple_plot
                # print(new__info)
                # print("\n-----------------------------------------------\n")
                # for key in new_info:
                # new_info = new_info[-4:]
                # print(Exception)
                #picked_info = pick_info(story, new__info, queries=queries)
                picked_info = new__info[-1][-1][-1]
                # print(picked_info)
                # print("\n-----------------------------------------------\n")
                new_info = []
                new_story = add_new_info(story, picked_info).replace("\n", " ").replace("  "," ")
                # print(new_story)
                # print("\n-----------------------------------------------\n")

                new_stories[index] = new_story
                direct_gen_stories[index] = story
               # w_retrieve_stories[index] = story
                picked_infos[index] = picked_info
                count+=1
            except Exception as e:
                print("ID:",index)
                print(str(e))

            with open("../data/ablation/" + ablation + "/new_stories.json", 'w') as json_file:
                json.dump(new_stories, json_file)
            with open("../data/ablation/" + ablation + "/direct_gen_stories.json", 'w') as json_file:
                json.dump(direct_gen_stories, json_file)
            # with open("../data/ablation/w_retrieve_stories.json", 'w') as json_file:
            #     json.dump(w_retrieve_stories, json_file)
            with open("../data/ablation/" + ablation + "/picked_info.json", 'w') as json_file:
                json.dump(picked_infos, json_file)
            print(index, "done")

    # with open("../data/ablation/new_stories.json",
    #           'w') as json_file:
    #     json.dump(new_stories, json_file)