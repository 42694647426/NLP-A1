
import nltk
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import string
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB

# load text data
df_neg = pd.read_csv('rt-polaritydata\\rt-polarity.neg',sep='\t', encoding='latin-1', header=None)
df_pos = pd.read_csv('rt-polaritydata\\rt-polarity.pos',sep='\t', encoding='latin-1', header=None)

df_neg.columns = ['Reviews']
df_pos.columns = ['Reviews']
df_neg['Sentiment'] = "neg"
df_pos['Sentiment'] = "pos"

df = pd.concat([df_neg, df_pos])
print(df[:4])


#remove punctuations 
def remove_punc(text):
    str = "".join([c for c in text if c not in string.punctuation])
    return str

df['Reviews'] = df['Reviews'].apply(lambda x: remove_punc(x))

#tokenize sentences
tokenizer = RegexpTokenizer(r'\w+')
df['Reviews'] = df['Reviews'].apply(lambda x: tokenizer.tokenize(x.lower()))

#remove stop words
def remove_stopwords(words):
    word = [w for w in words if w not in stopwords.words('english')]
    return word

df['Reviews'] = df['Reviews'].apply(lambda x: remove_stopwords(x))

#lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(word):
    lem = [lemmatizer.lemmatize(i) for i in word]
    return lem
df['Reviews'] = df['Reviews'].apply(lambda x: lemmatize_text(x))

#stemmatization
stemmer = PorterStemmer()
def stem_text(word):
    stem_word = [stemmer.stem(i) for i in word]
    return stem_word
df['Reviews'] = df['Reviews'].apply(lambda x: stem_text(x))
print(df[:5])

# shuffle the reviews 
df = shuffle(df)
word_pos= []
all_words = []
for s in df['Reviews']:
    word_pos.append(nltk.pos_tag(s))

print(word_pos[:5])
allowed_type = ["JJ", "JJR", "JJS","NN"]
words=[]

for i in word_pos:
    for j in i:
        words.append(j)
#print("words: ")
#print(words[:5])
for w in words:
        if w[1] in allowed_type:
            all_words.append(w[0])
#print("adjs: ")
#print(all_words[:5])


#count word frequencies 
all_words = nltk.FreqDist(all_words)

#select features 
features = list(all_words)[:500]
#print(features[:5])

#build feature set
def find_features(text):
    res = {}
    for w in features:
        res[w] = (w in text)
    return res

X_set = []
Y_set = []
for index, row in df.iterrows():
    X_set.append(find_features(row['Reviews']))
    Y_set.append(row['Sentiment']) 
featuresets = [(find_features(row['Reviews']), row['Sentiment']) for index, row in df.iterrows()]
#split test train set
#random.shuffle(featuresets)
train_set = featuresets[:7463]
test_set = featuresets[7463:]
#split train set test set
X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size = 0.3, random_state = True)
print(X_train[:10])

#fit naive bayes model

NaiveBayes = nltk.NaiveBayesClassifier.train(train_set)

print("Navie Bayes accuracy: ",nltk.classify.accuracy(NaiveBayes, test_set) )

NaiveBayes.show_most_informative_features(15)
#f1 = f1_score(labels=['neg', 'pos'], average='micro')
#print(f1)

clf = GaussianNB()
clf.fit(X_train, Y_train)

plot_confusion_matrix(clf, X_test, Y_test)  
plt.show() 

#all_words = [(R, S) for R, S in df.iterrows()]

#cv = CountVectorizer()
#text_count = cv.fit_transform(df['Reviews'])



