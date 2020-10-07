import nltk
import pandas as pd
import matplotlib
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
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from wordcloud import WordCloud
import cv2 
# load text data
df_neg = pd.read_csv('rt-polaritydata\\rt-polarity.neg',sep='\t', encoding='latin-1', header=None)
df_pos = pd.read_csv('rt-polaritydata\\rt-polarity.pos',sep='\t', encoding='latin-1', header=None)

df_neg.columns = ['Reviews']
df_pos.columns = ['Reviews']
df_neg['Sentiment'] = "neg"
df_pos['Sentiment'] = "pos"

df = pd.concat([df_neg, df_pos])
print("Original dataset: ")
print(df[:4])

#remove punctuations 
def remove_punc(text):
    str = "".join([c for c in text if c not in string.punctuation])
    return str

df['Reviews'] = df['Reviews'].apply(lambda x: remove_punc(x))

#tokenize sentences
tokenizer = RegexpTokenizer(r'\w+')
df['Reviews'] = df['Reviews'].apply(lambda x: tokenizer.tokenize(x.lower()))

unprocessed_df = df


#plot cloud graph
df_neg['Reviews']= df_neg['Reviews'].apply(lambda x: remove_punc(x))
df_pos['Reviews']= df_pos['Reviews'].apply(lambda x: remove_punc(x))

text_neg = ' '.join(df_neg['Reviews'])
text_pos = ' '.join(df_pos['Reviews'])
wordcloud_neg = WordCloud(background_color='white',max_font_size = 50,stopwords=stopwords.words('english'),
        max_words=150,
        scale=3,
        random_state=1).generate(text_neg)
wordcloud_pos = WordCloud(background_color='white',max_font_size = 50,stopwords=stopwords.words('english'),
        max_words=150, 
        scale=3,
        random_state=1).generate(text_pos)
plt.figure(figsize = (15, 9))
# Display the generated image:

#plt.imshow(wordcloud_neg,interpolation= "bilinear")
plt.imshow(wordcloud_pos,interpolation= "bilinear")
plt.axis("off")
plt.show()


# remove punctions and tokenize sentiences are basic preprocessing steps, then test if we need to remove stop words
#lemmatize, stem, words. et. using pipline. 
#remove stop words
def remove_stopwords(words):
    word = [w for w in words if w not in stopwords.words('english')]
    return word
#df['Reviews'] = df['Reviews'].apply(lambda x: remove_stopwords(x))

#lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(word):
    lem = [lemmatizer.lemmatize(i) for i in word]
    return lem
#df['Reviews'] = df['Reviews'].apply(lambda x: lemmatize_text(x))

#stemmatization
stemmer = PorterStemmer()
def stem_text(word):
    stem_word = [stemmer.stem(i) for i in word]
    return stem_word
#df['Reviews'] = df['Reviews'].apply(lambda x: stem_text(x))

# automatize the preprocessing
def processing(df, stem=True, lemma = True, stop = True):
    if(stem):
        df['Reviews'] = df['Reviews'].apply(lambda x: stem_text(x))
    if(lemma):
        df['Reviews'] = df['Reviews'].apply(lambda x: lemmatize_text(x))
    if(stop):
        df['Reviews'] = df['Reviews'].apply(lambda x: remove_stopwords(x))
    return(df)


#build feature set
def find_features(text, features):
    res = {}
    for w in features:
        res[w] = (w in text)
    return res

def feature_set(df, N=1000): #N = number of features to be selected
    # shuffle the reviews 
    df = shuffle(df)
    word_pos= []
    all_words = []
    for s in df['Reviews']:
        word_pos.append(nltk.pos_tag(s)) # add tags to every word
    allowed_type = ["JJ", "JJR", "JJS","NN", 
    "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN"] #allow adjectives, Nouns and verbs not adverbs, conjunctions 
    words=[] #unlist the words array
    for i in word_pos:
        for j in i:
            words.append(j)
    for w in words:
        if w[1] in allowed_type:
            all_words.append(w[0])
    all_words = nltk.FreqDist(all_words) #count frequency of words
    features = list(all_words)[:N] # extract the 500 most frequent words as features
    featuresets = [(find_features(row['Reviews'], features), row['Sentiment']) for index, row in df.iterrows()]
    return featuresets

#split train set test set
'''
X_set = []
Y_set = []
for index, row in df.iterrows():
    X_set.append(find_features(row['Reviews'], features))
    Y_set.append(row['Sentiment']) 
X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size = 0.3, random_state = True)
print(X_train[:10])
'''

def test_preprocess(df = unprocessed_df, stem=True, lemma = True, stop = True):
    df = processing(df, stem, lemma, stop)
    featuresets = feature_set(df)
    random.shuffle(featuresets)
    train_set = featuresets[:7463]
    test_set = featuresets[7463:]

    #Naive Bayes
    NaiveBayes = nltk.NaiveBayesClassifier.train(train_set)
    
    best_acc=nltk.classify.accuracy(NaiveBayes, test_set)
    best_model = "Naive Bayes"
    print("Navie Bayes accuracy: ",best_acc)
    NaiveBayes.show_most_informative_features(15)

    #Logistic regression 
    LogReg_clf = SklearnClassifier(LogisticRegression())
    LogReg_clf.train(train_set)
    LogReg_acc = nltk.classify.accuracy(LogReg_clf, test_set)
    if(LogReg_acc>best_acc):
        best_acc = LogReg_acc
        best_model = "Logistic Regression"
    print("Logistic Regression accuracy: ", LogReg_acc)

    #support vector machine
    SVM_clf = SklearnClassifier(SVC())
    SVM_clf.train(train_set)
    SVM_acc = nltk.classify.accuracy(SVM_clf, test_set)
    if(SVM_acc>best_acc):
        best_acc = SVM_acc
        best_model = "Support Vector Machine"
    print("Support Vector Machine accuracy: ", SVM_acc)

    #MLP Neural Network
    DT_clf = SklearnClassifier(DecisionTreeClassifier())
    DT_clf.train(train_set)
    DT_acc = nltk.classify.accuracy(DT_clf, test_set)
    if(DT_acc>best_acc):
        best_acc = DT_acc
        best_model = "Decision Tress"
    print("Decision Tree accuracy: ", DT_acc)

    print("Best model is", best_model, ", Best accuracy: ", best_acc)
    return best_acc


def run_test(df =unprocessed_df ):
    l =[]
    dict={'stem': False, 'lemma': False, 'stop':False }
    print("Preprocessing methods:", dict)
    unprocess = test_preprocess(df, stem = False, lemma = False, stop = False)
    l.append((dict,unprocess))

    print("------------------------------------------------")
    dict={'stem': True, 'lemma': True, 'stop':True }
    print("Preprocessing methods:", dict)
    origin = test_preprocess(df)
    l.append((dict,origin))

    print("------------------------------------------------")
    dict={'stem': False, 'lemma': False, 'stop':False }
    dict['stem'] = True
    print("Preprocessing methods:", dict)
    stem = test_preprocess(df, lemma = False, stop = False)
    l.append((dict,stem))
    dict['stem'] = False

    print("------------------------------------------------")
    dict['lemma'] =True
    print("Preprocessing methods:", dict)
    lemma = test_preprocess(df, stem = False, stop = False)
    l.append((dict,lemma))
    dict['lemma'] = False

    print("------------------------------------------------")
    dict['stop'] =True
    print("Preprocessing methods:", dict)
    stop = test_preprocess(df,stem = False, lemma = False)
    l.append((dict,stop))
    dict[2] = False
    return l

print(run_test())

# plot confusion matrix
df = processing(df)
print("Processed dataset: ")
print(df[:5])

featuresets = feature_set(df)
#split test train set
random.shuffle(featuresets)
train_set = featuresets[:7463]
test_set = featuresets[7463:]


def plot_confusion_matrix(y_true, y_pred, classes, classifier = "Naive Bayes",
                          cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix", classifier)

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    #show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Normalized confusion matrix for ' + classifier,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.5f'  
    thresh = cm.max() / 2+0.1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=10)

NaiveBayes = nltk.NaiveBayesClassifier.train(train_set)
print("Navie Bayes accuracy: ",nltk.classify.accuracy(NaiveBayes, test_set))
NaiveBayes.show_most_informative_features(15)

ground_truth = [r[1] for r in test_set]
preds = [NaiveBayes.classify(r[0]) for r in test_set]

y_test = ground_truth
y_pred = preds
class_names = ['neg', 'pos']

plot_confusion_matrix(y_test, y_pred, classes=class_names)

#logistic regression
LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(train_set)
LogReg_acc = nltk.classify.accuracy(LogReg_clf, test_set)
print("Logistic Regression accuracy: ", LogReg_acc)

preds = [LogReg_clf.classify(r[0]) for r in test_set]
y_pred = preds


plot_confusion_matrix(y_test, y_pred, classes=class_names, classifier="Logistic Regression")
#support vector machine
SVM_clf = SklearnClassifier(SVC())
SVM_clf.train(train_set)
SVM_acc = nltk.classify.accuracy(SVM_clf, test_set)
print("Support Vector Machine accuracy: ", SVM_acc)

preds = [SVM_clf.classify(r[0]) for r in test_set]
y_pred = preds


plot_confusion_matrix(y_test, y_pred, classes=class_names, classifier="Support Vector Machine")

#MLP Neural Network
DT_clf = SklearnClassifier(DecisionTreeClassifier())
DT_clf.train(train_set)
DT_acc = nltk.classify.accuracy(DT_clf, test_set)
print("Decision Tree accuracy: ", DT_acc)

preds = [DT_clf.classify(r[0]) for r in test_set]
y_pred = preds

plot_confusion_matrix(y_test, y_pred, classes=class_names, classifier="Decision Tree")

plt.show()


