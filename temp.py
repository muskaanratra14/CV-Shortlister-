import slate3k as slate
import re
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
escape_char = re.compile(r'\\x[0123456789abcdef]+')
resume_list = []
labelList=[]

def remove_punctuation_marks(resumeString):
    withoutPunct = ""
    for ch in resumeString:
        if ch not in punctuations:
            withoutPunct = withoutPunct + ch
    return withoutPunct

def remove_new_lines(resumeString):
    withoutNewLines = resumeString.replace('\\n','')
    return withoutNewLines
    
def process_resume_list():
    for resumeNo in range (1,98):
        resume = 'C:/Users/Muskaan Ratra/Desktop/CVs/CVs/c' + str(resumeNo) + '.pdf'

        resumeFile=open(resume,'rb')
        resumePdf = slate.PDF(resumeFile)       
        
        # Remove punctuaton marks
        removeNewLines = remove_new_lines(str(resumePdf))
        
        # Remove escape chars
        escapeCharsString = re.sub(escape_char, " ", removeNewLines)
        
        # Remove punctuation marks 
        finalString = remove_punctuation_marks(escapeCharsString)
        
        resume_list.append(finalString)
        
    
    # Start lablabel=[]
    for i in range(36):
        labelList.append(1)
    for i in range(61):
        labelList.append(0)
    print(np.array(labelList))
        
    
def main():
    process_resume_list()


save_model = 'finalized_model.sav'
save_vector = 'finalized_vectorizer.sav'

if __name__ == '__main__':
    main()

    label=np.array(labelList)
    
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',max_features=250)
    resumes_train,resumes_test,y_train,y_test=train_test_split(resume_list,label,test_size=0.33,random_state=1)
    X_train = vectorizer.fit_transform(resumes_train)
    X_test = vectorizer.fit_transform(resumes_test)
    
    X_train_array = X_train.toarray()
    X_test_array  = X_test.toarray()
    y_test1=y_test.reshape(-1,1)
    
    
    print(vectorizer.get_feature_names())
    pickle.dump(vectorizer, open(save_vector, 'wb'))
    
    #Implementing Bernoulli Naive Bayes
    naive_bayes = BernoulliNB(alpha=1.0)
    naive_bayes.fit(X_train_array, y_train)
    predictions = naive_bayes.predict(X_test_array)
    naivescore=(naive_bayes.score(X_test_array, y_test1))*100
    
    #Implementing Guassian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train_array, y_train)
    y_pred = gnb.predict(X_test_array)
    gnbscore = (gnb.score(X_test_array,y_test1))*100
    
    
    dtree_model = DecisionTreeClassifier(max_depth=2)
    dtree_model.fit(X_train_array, y_train) 
    dtree_predictions = dtree_model.predict(X_test_array) 
    dtscore=(dtree_model.score(X_test_array, y_test1))*100
    
    model = SVC(kernel='linear')
    model.fit(X_train_array, y_train)
    model_pred=model.predict(X_test_array)
    svcscore=(model.score(X_test_array,y_test1))*100
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_array,y_train)
    clf_pred=clf.predict(X_test_array)
    clfscore=(clf.score(X_test_array,y_test1))*100
      
    knn = KNeighborsClassifier(n_neighbors = 11,metric='minkowski' , p=2).fit(X_train_array, y_train) 
    knnscore=(knn.score(X_test_array,y_test1))*100
    
    scores = [gnbscore,naivescore,svcscore,knnscore,dtscore,clfscore]
    algorithms = ["Gaussian naive bayes","Bernoulli naive bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]
    sns.set(rc={'figure.figsize':(15,8)})
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")
    sns.barplot(algorithms,scores)
    
    final_model = naive_bayes
    
    # save the model to disk
    pickle.dump(final_model, open(save_model, 'wb'))
    
def make_prediction(resumeNo):
    resume = 'C:/Users/Muskaan Ratra/Desktop/CVs/CVs/c' + str(resumeNo+1) + '.pdf'
    loaded_model = pickle.load(open(save_model, 'rb'))
    loaded_vector = pickle.load(open(save_vector, 'rb'))
    resumeFile=open(resume,'rb')
    sample_resume=slate.PDF(resumeFile)
    sample_resume=sample_resume[0]
    sample_resume=loaded_vector.transform([sample_resume])
    return loaded_model.predict(sample_resume)[0]
    
#print(make_prediction(4))