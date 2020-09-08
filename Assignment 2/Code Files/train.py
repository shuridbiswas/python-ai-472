# -------------------------------------------------------
# Assignment 2
# Written by Shurid Biswas 40024592
# For COMP 472 Section ABKX – Summer 2020
# --------------------------------------------------------
import pandas as pd  #pandas library use for reading dataset
import nltk          #The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language
import experiments   #The experiment module provides a rich API for running versioned Machine Learning experiments, whether it be a simple single-process Python application or Distributed Training over many machines.
import time          #time library used for calculating time of code
import operator      #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python. 
from collections import OrderedDict  #OrderedDict preserves the order in which the keys are inserted. A regular dict doesn’t track the insertion order, and iterating it gives the values in an arbitrary order. By contrast, the order the items are inserted is remembered by OrderedDict.
import xlsxwriter    #used for writing excel file
import warnings         #Python programmers issue warnings by calling the warn() function defined in this module. 
#import matplotlib inline
warnings.filterwarnings('ignore') # Python programmers issue warnings by calling the warn() function defined in this module. 
remove_freq = 1
remove_percent = 0
smoothing_value = 0


nltk.download('averaged_perceptron_tagger') #download and install nltk data average_perception_tagger
nltk.download('wordnet') #download and install nltk data wordnet 


def read_file(exp=1):  #creating function for read file
    global df_testing  #declearing global variable for testing
    global df_training  #declearing global variable for training

    df = pd.read_csv("hns_2018_2019.csv") #read data file using pandas library hns is excel dataset
    # df = pd.read_csv("./sample.csv")
    df = df.drop(columns=df.columns[0]) #drop coloumn on 0 index
    df['date'] = pd.to_datetime(df['Created At']) #formating date in column
    start_date = '2018-01-01 00:00:00' #start date define in excel column
    end_date = '2018-12-31 00:00:00'  #end date define in excel column
    mask_2018 = (df['date'] > start_date) & (df['date'] <= end_date) #define condition on date column that we are used
    start_date = '2019-01-01 00:00:00' #start date define in excel column
    end_date = '2019-12-31 00:00:00' #end date define in excel column
    mask_2019 = (df['date'] > start_date) & (df['date'] <= end_date)#define condition on date column that we are used
    df_training = df.loc[mask_2018] #assign data for training
    df_testing = df.loc[mask_2019]  #assign data for testing
    build_vocabulary(df_training, exp) #passing parameter for vocabulary function


def build_vocabulary(df, exp): ##creating a vocabulary function 
    
    global word_freq_dict #define global variable 
    global start_time #define global variable
    global words_removed #define global variable
    word_freq_dict = {} #define dictionary
    words_removed = set() #creating a tuple 

    start_time = time.process_time() #used time library for calculating time 

    if exp == 2: #creating a if condition if exp==2 then 
        stop_words_df = pd.read_csv("stopwords.txt") #read data in stopword txt file
        stop_words = stop_words_df["a"].tolist()

    for index, row in df.iterrows():
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True) #A tokenizer that divides a string into substrings by splitting on the specified string (defined in subclasses).

        raw = tokenizer.tokenize(row["Title"].lower()) #Convert title row into lower case

        temp1 = tokenizer.tokenize(row["Title"]) #store data into temporary variable

        title = ' '.join(temp1)#The join() method is a string method and returns a string in which the elements of sequence have been joined by str separator.


        if exp == 2: #check if condition  if exp==2 then perform this function
            raw = list(set(raw).difference(stop_words)) 
            row["Title"] = ' '.join([str(elem) for elem in raw])
        elif exp == 3: #check if condition==3 then perform this line
            for each in raw: #using for loop perform on raw variable
                if len(each) >= 9 or len(each) <= 2: #check condition if length of each >=9 nad <=2 
                    raw.remove(each) #the remove in raw

        tokenize_word(raw, title, df, index, words_removed)  # tokenizer function call in this line. A tokenizer that divides a string into substrings by splitting on the specified string (defined in subclasses).

    od = OrderedDict(sorted(word_freq_dict.items())) #call orderdict and sorted values in word_freq

    with open('frequency_dict.txt', 'w' ,encoding='utf-8') as file: #using files write file in folder with frequency_dict name  
        for key, val in od.items(): #using for loop for accessing values in files
            file.write(str(key) + " " + str(val) + "\n") #Write command is used for writing data in a file

    with open("./remove_word.txt", "w" ,encoding='utf-8') as file: #write file in a folder with remove_word.txt name
        for element in words_removed: #using for loop for accessing files values or file words
            file.write(element + "\n") #write data in a file

    train(od, exp) #traing module used in this location

    total_time = time.process_time() - start_time #time libarary used for calculting time 
    print('TOTAL TIME TAKEN IN (S):', total_time) #print total time of command in seconds
    print('TOTAL TIME TAKEN IN (MINUTES):', total_time / 60) #print total time in a minute
    print("-------------------------------------------------") #print ----


def tokenize_word(raw, title, df, index, w_removed, testing=False): #defing a new function 
    bigrams = [] #making a list 
    word_list = [] #making a list

    lemmatizer = nltk.WordNetLemmatizer() #Lemmatization is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. 

    bigrm = list(nltk.bigrams(title.split()))
    pos = nltk.pos_tag(raw) #POS-tagger, processes a sequence of words, and attaches a part of speech tag to each word
    pos_dict = dict(pos) #assign pos to dictionary

    for i in bigrm: #used for loop in bigrm variable
        bigrams.append((''.join([w + ' ' for w in i])).strip()) #append data in a list

    for each_element in bigrams: #for loop used in each element of bigrams
        word = each_element.split(' ') #split word by space

        indices_0 = [i for i, e in enumerate(raw) if e == word[0].lower()] #apply for loop and if condition and then converting  data into lowercase
        if len(indices_0) != 0: #check condition of len of index is not equal to zero then exeeuate this line
            indices_1 = [i for i, e in enumerate(raw[indices_0[0] + 1:]) if e == word[1].lower()] #perform for loop on index 1 and then check condition and then converting all word into lower case
        else: #if condition false then execute this line
            indices_1 = [i for i, e in enumerate(raw) if e == word[1].lower()] #check index 1 and An enumeration defines a common type for a group of related values and enables you to work with those values in a type-safe way within your code.

        if word[0].istitle() and word[1].istitle():
            if len(indices_0) > 0 and (
                    pos_dict.get(word[0].lower()) == 'NN' or pos_dict.get(word[0].lower()) == 'NNS'):
                if len(indices_1) > 0 and (
                        pos_dict.get(word[1].lower()) == 'NN' or pos_dict.get(word[1].lower()) == 'NNS'):
                    raw.remove(word[0].lower())
                    # raw.index(word[1].lower())
                    raw.remove(word[1].lower())

                    if testing is False: #check true or false condition if true then execute this line 
                        temp = each_element.lower() + "-" + df.at[index, 'Post Type'] # store data in temporary variable
                        freq = word_freq_dict.get(temp) #get word frequency
                        raw.append(each_element.lower()) #append data in raw 
                        if freq is None: #checking if frequency is null or nothing then 
                            word_freq_dict[temp] = 1 # assign vlaue by 1
                        else: #if condition false then
                            freq += 1 #increment frequency by 1
                            word_freq_dict[temp] = freq
                    else:
                        word_list.append(each_element.lower()) #append data into wordlist 

    pos = nltk.pos_tag(raw)#function clll ntlk

    for each_word in pos: #for loop onf pos 
        wordnet_tag = get_wordnet_pos(each_word[1]) #checking condition by index

        if each_word[1] == "FW" or each_word[1] == "CD": # apply condition if word start with FW or CD Then
            w_removed.add(each_word[0].strip()) #The strip() method removes characters from both left and right based on the argument (a string specifying the set of characters to be removed).
            continue #if true then skip this command 
        if len(each_word[0]) == 1 and not (each_word[0] == "a" or each_word[0] == "i"):
            w_removed.add(each_word[0].strip()) #The strip() method removes characters from both left and right based on the argument (a string specifying the set of characters to be removed).
            continue #if true then skip this command and again run loop loop

        word_lemm = lemmatizer.lemmatize(each_word[0], wordnet_tag) #calling function 

        if testing is False: #checking true or false condition
            temp = word_lemm + "-" + df.at[index, 'Post Type'] # store in a temporary variable 
            value = word_freq_dict.get(temp) #assign tempprary variable to value 
            if value is None: #check condtition if value is none or empty 
                word_freq_dict[temp] = 1 #then assign value by 1
            else: #if condition is false then
                value += 1 #increment value by 1
                word_freq_dict[temp] = value #store value in a list
        else: #if condition is false 
            if testing: #check nested condition 
                word_list.append(word_lemm) #append value in a word_list

    pos.clear() #clear data
 
    return word_list #return world_list


def train(freq_dict, exp): #define train function 
    word = [] #making list 
    word_list = [] #making list 
    post_type = [] #making list 
    p_ask_hn_dict = {} #making Dictionary for storing data
    p_story_dict = {} #making Dictionary for storing data
    p_show_hn_dict = {} #making Dictionary for storing data
    p_poll_dict = {} #making Dictionary for storing data
    class_probability = [] #making list 
    smoothing = 0.5 #set thershold 0.5

    if exp == 4:
        new_dict = {k: v for k, v in freq_dict.items() if not (v <= remove_freq)}
        freq_dict = new_dict
    elif exp == 4.5:
        sorted_dict_list = sorted(freq_dict.items(), key=operator.itemgetter(1))
        remove_elements = int(len(sorted_dict_list) * remove_percent)
        new_dict_list = sorted_dict_list[remove_elements:]
        freq_dict = dict(new_dict_list)
    elif exp == 5:
        smoothing = smoothing_value

    dict_keys = freq_dict.keys()
    freq = list(freq_dict.values())

    for each in dict_keys:
        word_class = each.split('-')
        word.append(word_class[0])
        post_type.append(word_class[1])

    df = pd.DataFrame({'Word': word, 'Class': post_type, 'Frequency': freq})
    df.to_csv("vocabulary.csv")

    if df.empty:
        experiments.each_accuracy = -1
        return

    story_df = df[df.Class.str.match('story', case=False)]
    ask_hn_df = df[df.Class.str.match('ask_hn', case=False)]
    show_hn_df = df[df.Class.str.match('show_hn', case=False)]
    poll_df = df[df.Class.str.match('poll', case=False)]

    story_dft = df_training[df_training["Post Type"].str.match('story', case=False)]
    ask_hn_dft = df_training[df_training["Post Type"].str.match('ask_hn', case=False)]
    show_hn_dft = df_training[df_training["Post Type"].str.match('show_hn', case=False)]
    poll_dft = df_training[df_training["Post Type"].str.match('poll', case=False)]

    show_hn_words = dict(zip(show_hn_df.Word, show_hn_df.Frequency))
    ask_hn_words = dict(zip(ask_hn_df.Word, ask_hn_df.Frequency))
    poll_words = dict(zip(poll_df.Word, poll_df.Frequency))
    story_words = dict(zip(story_df.Word, story_df.Frequency))

    show_hn_count = sum(show_hn_words.values())
    ask_hn_count = sum(ask_hn_words.values())
    poll_count = sum(poll_words.values())
    story_count = sum(story_words.values())

    vocabulary = df.Word.unique()
    vocabulary_size = len(vocabulary)
    experiments.no_of_words = vocabulary_size

    class_probability_show_hn = len(show_hn_dft.index) / len(df_training.index)
    class_probability_ask_hn = len(ask_hn_dft.index) / len(df_training.index)
    class_probability_poll = len(poll_dft.index) / len(df_training.index)
    class_probability_story = len(story_dft.index) / len(df_training.index)

    if smoothing == 0:
        vocabulary_size = 0

    line_count = 1

    for word in vocabulary:
        temp_show_hn_freq = show_hn_words[word] if word in show_hn_words else 0
        temp_ask_hn_freq = ask_hn_words[word] if word in ask_hn_words else 0
        temp_story_freq = story_words[word] if word in story_words else 0
        temp_poll_freq = poll_words[word] if word in poll_words else 0

        if show_hn_count == 0:
            p_word_given_show_hn = 0
        else:
            p_word_given_show_hn = ((temp_show_hn_freq + smoothing) / (show_hn_count + vocabulary_size))

        if ask_hn_count == 0:
            p_word_given_ask_hn = 0
        else:
            p_word_given_ask_hn = ((temp_ask_hn_freq + smoothing) / (ask_hn_count + vocabulary_size))

        if poll_count == 0:
            p_word_given_poll = 0
        else:
            p_word_given_poll = ((temp_poll_freq + smoothing) / (poll_count + vocabulary_size))

        if story_count == 0:
            p_word_given_story = 0
        else:
            p_word_given_story = ((temp_story_freq + smoothing) / (story_count + vocabulary_size))

        if exp == 1: #if condition is true   
            file = open("model-2018.txt", "a",encoding='utf-8') #append file in a folder by name model-2018
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str( #perform opertion on file such as line count in a file 
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close() #close file 
        elif exp == 2:  #if condition is true 
            file = open("stopword-model.txt", "a",encoding='utf-8') #append or create new file in a folder by name stopword.txt
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str( #file write operations
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close() #  close file 
        elif exp == 3:#if condition is true 
            file = open("wordlength-model.txt", "a",encoding='utf-8')  #append or create new file in a folder by name wordlength.txt
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str(  #file write operations
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close() #close file 
        line_count += 1 #increment by 1  or count line

        p_ask_hn_dict[word] = p_word_given_ask_hn
        p_show_hn_dict[word] = p_word_given_show_hn
        p_poll_dict[word] = p_word_given_poll
        p_story_dict[word] = p_word_given_story
        word_list.append(word)

    end_time = time.process_time() - start_time
    print("\nTime to train:", end_time)

    # 0: show_hn
    # 1: ask_hn
    # 2: poll
    # 3: story

    class_probability.append(class_probability_show_hn)
    class_probability.append(class_probability_ask_hn)
    class_probability.append(class_probability_poll)
    class_probability.append(class_probability_story)

    accuracy = experiments.baseline(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
                                    p_story_dict, exp)

    if exp == 4 or exp == 4.5 or exp == 5:
        experiments.each_accuracy = accuracy


def get_wordnet_pos(treebank_tag): #defining function for geting wordnet
    if treebank_tag.startswith('J'): #condition check if treebank start with J then 
        return 'a' #return a 
    elif treebank_tag.startswith('V'): #condition check if treebank start with J then 
        return 'v'#return v
    elif treebank_tag.startswith('N'): #condition check if treebank start with J then 
        return nltk.wordnet.NOUN
    elif treebank_tag.startswith('R'): #condition check if treebank start with J then 
        return 'r'#return noun 
    else:
        return nltk.wordnet.NOUN#return noun


if __name__ == '__main__': #calling main function
    experiments.select_experiment() #calling experiment module and then select experiment function