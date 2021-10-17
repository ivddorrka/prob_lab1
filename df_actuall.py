import pandas as pd
import re

def read_stopwords(path: str) -> set:
    """
    Function for reading data from file.
    :param data_file: str - stopwords data
    :return: set of words
    """
    with open(path) as file:
        stopwords = file.read().splitlines()
    return set(stopwords)

def process_data(path: str) -> tuple:
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data do a research of your own
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    df = pd.read_csv (path)

    all_words, good, bad = [], [], []
    num_neutral = 0
    num_bad = 0
    stopwords = read_stopwords("stop_words.txt")

    for tweet, label in zip(df["tweet"], df["label"]):
        sentence = list(filter(None, re.sub("[^A-Za-z ]", "", tweet).strip().split(' ')))
        cleaned_sentence = list(filter(lambda x: x not in stopwords, sentence))
        if label=='neutral':
            for word in range(len(cleaned_sentence)):
                good.append(cleaned_sentence[word])
                if cleaned_sentence[word] not in all_words:
                    all_words.append(cleaned_sentence[word])
            num_neutral += 1
        else:
            for word in range(len(cleaned_sentence)):
                bad.append(cleaned_sentence[word])
                if cleaned_sentence[word] not in all_words:
                    all_words.append(cleaned_sentence[word])
            num_bad += 1
        


    dict_1 = get_lst_01(good, bad, all_words)
    
    return dict_1, good, bad, len(all_words), num_bad, num_neutral



def get_lst_01(good, bad, all_words):
    """Dictionary returns tuple of number of times whish word appears in good, bad and the total amount of times"""
    dict_1 = {}
    for i in range(len(all_words)):
        num_in_good = good.count(all_words[i])
        num_in_bad = bad.count(all_words[i])
        sum_num_word=num_in_bad+num_in_good
        dict_1[all_words[i]]= [num_in_good, num_in_bad, sum_num_word]
    return dict_1




class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        self.data = process_data('train.csv')
        self.dict_words = self.data[0]
        self.good = self.data[1]
        self.bad = self.data[2]
        self.stopwords = read_stopwords("stop_words.txt")
        self.number_word = self.data[3]
        self.num_bad = self.data[4]
        self.num_neutral= self.data[5]


    def fit(self, X, y):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        pass

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        sentence = list(filter(None, re.sub("[^A-Za-z ]", "", message).strip().split(' ')))
        cleaned_sentence = list(filter(lambda x: x not in self.stopwords, sentence))
        result_probability = 1
        if label=='discrim':
            result_probability = result_probability*((self.num_bad)/(self.num_neutral+self.num_bad))
            for i in cleaned_sentence:
                dict_result = self.dict_words.get(i)
                if dict_result != None:
                    number_bad = dict_result[1]+1
                    all_words = dict_result[3]
                else:
                    number_bad = 1
                    all_words = self.number_word + 1
                result_probability = result_probability*(number_bad/(len(self.bad)+all_words))
        

        if label=='neutral':
            result_probability = result_probability*((self.num_neutral)/(self.num_neutral+self.num_bad))
            # all_words = 0
            for i in cleaned_sentence:
                dict_result = self.dict_words.get(i)
                if dict_result:
                    number_good = dict_result[0]+1
                    all_words = dict_result[3]
                else:
                    number_good = 1
                    all_words =  self.number_word + 1
                result_probability = result_probability*(number_good/(len(self.good)+all_words))
        return result_probability

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        number_good = self.predict_prob(message, 'neutral')
        number_bad = self.predict_prob(message, 'discrim')
        if number_bad>number_good:
            return 'discrim'
        return 'neutral'


    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        print('got here')
        count = 0
        for i in range(len(X)):
            lab = self.predict(X[i])
            if lab==y[i]:
                count += 1
        return (count/(len(y)))*100



def re_pro_cess(file):
    """ To get X and Y for the class score method"""
    df = pd.read_csv (file)
    stopwords = read_stopwords("stop_words.txt")
    lst_first = []
    lst_second = []
    for tweet, label in zip(df["tweet"], df["label"]):
        sentence = list(filter(None, re.sub("[^A-Za-z ]", "", tweet).strip().split(' ')))
        cleaned_sentence = ' '.join(list(filter(lambda x: x not in stopwords, sentence)))
        lst_first.append(cleaned_sentence)
        lst_second.append(label)
    return lst_first, lst_second


sd = re_pro_cess('test.csv')
c_o = BayesianClassifier()
print(c_o.score(sd[0], sd[1]))