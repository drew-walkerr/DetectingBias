## Script written by Jingfeng Yang
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import spacy
import matplotlib.pyplot as plt

def extract_entity(file, nlp):
    entities = {}
    with open(os.path.join('test1', file), 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            if line_num <2:
                line_num+=1
                continue
            sents = sent_tokenize(line.strip())
            for sent in sents:
                doc = nlp(sent)
                for ent in doc.ents:
                    entities[sent[ent.start_char: ent.end_char]] = ent.label_

    with open(os.path.join('entity', file), 'w', encoding='utf-8') as f:
        for k, v in entities.items():
            f.write(k + ' ' + v +'\n')

def extract_dep(file, nlp, freq_dic):
    adjs = []
    cur_year = 0
    with open(os.path.join('test1', file), 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            if line_num <2:
                if line_num==1:
                    cur_year=int(line[:4])
                line_num+=1
                continue

            sents = sent_tokenize(line.strip())
            for sent in sents:
                if 'immigrant' in word_tokenize(sent) or 'immigrants' in word_tokenize(sent):
                    #print(sent)
                    doc = nlp(sent)
                    for possible_imm in doc:
                        if possible_imm.dep_ == "amod" and (possible_imm.head.text == 'immigrant' or possible_imm.head.text == 'immigrants'):
                            adjs.append(possible_imm.text)
                            #print(possible_imm.text)
    if len(adjs)>0:
        if cur_year not in freq_dic:
            freq_dic[cur_year]={}
        for adj in adjs:
            if adj not in freq_dic[cur_year]:
                freq_dic[cur_year][adj]=1
            else:
                freq_dic[cur_year][adj]+=1

    with open(os.path.join('immAdj', file), 'w', encoding='utf-8') as f:
        for a in adjs:
            f.write(a+'\n')

def plot_freq(freq_dic):
    year_list = set() #whats the difference between this and []
    key_list = set()
    for year, terms in freq_dic.items():
        year_list.add(year)
        total_num=0
        for k, v in terms.items():
            total_num+=v
            key_list.add(k)
        for k in terms.keys():
            terms[k] = terms[k]/total_num

    year_list = list(year_list)
    year_list.sort()

    print(len(key_list))

    for term in list(key_list):
        freq_list=[]
        occured_year_numbers = 0
        for year in year_list:
            if term in freq_dic[year]:
                freq_list.append(freq_dic[year][term])
                occured_year_numbers+=1
            else:
                freq_list.append(0.0)
        if occured_year_numbers>5:
            plt.plot(year_list,freq_list,'o-',label=term)
    plt.xlabel("year")
    plt.ylabel("freq")
    plt.legend(loc = "best")
    plt.show()




if __name__ == "__main__":
    sents = []
    entity_nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    dep_nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner"])
    freq_dic={}
    for filename in os.listdir('test1'):
        #extract_entity(filename, entity_nlp)
        extract_dep(filename, dep_nlp, freq_dic)
    plot_freq(freq_dic)
