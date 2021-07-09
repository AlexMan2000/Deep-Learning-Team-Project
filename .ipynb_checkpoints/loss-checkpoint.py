import spacy
from numpy import sqrt
import numpy as np

nlp=spacy.load('en_core_web_md')

class loss:
    def __init__(self,sentence,n):
        self.n = n
        self.doc= list()
        for i in range(n):
            self.doc.append(nlp(sentence[i]).vector)

    def distance(self,other,distance):
        if distance=='Euclidean':
            result=list()
            for i in range(self.n):
                result.append(self.doc[i]-other.doc[i])
            for j in range(self.n):
                result[j]=sum([i*i for i in result[j]])
            result=np.array(result)
            return sqrt(result)
        elif distance=='Manhattan':
            result=list()
            for i in range(self.n):
                result.append(self.doc[i]-other.doc[i])
            for j in range(self.n):
                result[j]=sum([abs(i) for i in result[j]])
            result=np.array(result)
            return result
        
        

if __name__=='__main__':
    doc=loss(['I\'m a man','I\'m a woman'],2)
    doc1=loss(['I am a man','asjkdhflkjsahfskldjhfalskjfhasdlkjfhlsdkajf'],2)
    print(doc.distance(doc1,'Euclidean'))
    print(doc.distance(doc1,'Manhattan'))