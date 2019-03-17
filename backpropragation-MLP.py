# Dandi Wiratsangka

import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class NNetwork:

    def __init__(self,ni,nh,no,alpha,epoch,wih=None,who=None):
        """
        ni : number of neuron pada input layer
        nh : --------------- pada hidden layer
        no : ---------------- pada output layer
        wih : weight dari input ke hidden layer
        who : weight dari hidden ke output layer
        """
        self.ni = ni
        self.nh = nh
        self.no = no

        self.LEARNING_RATE =  alpha
        self.EPOCH = epoch  
        
        self.wih=[w[:] for w in wih] if (wih is not None) else [[0.1 for i in range(self.ni+1)] for j in range(self.nh)]                
        self.who=[w[:] for w in who] if (who is not None) else [[0.1 for i in range(self.nh+1)] for j in range(self.no)]         
        
        self.train_errors = [] # size : EPOCH x jumlah trainset
        self.train_accuracy = []
        self.validation_errors = []
        self.validation_accuracy = []

    def forwardPass(self,inputs,wih,who):
        """
        menghitung dan mengembalikan nilai aktivasi (output) dari hidden dan output layer
        oh : output of hidden layer
        oo : output of output layer
        """                
        netsHidden=[self.netResult(inputs,wih[i]) for i in range(self.nh)]        
        oh = self.sigmoid(netsHidden)        
        netsOuput = [self.netResult(oh,who[i]) for i in range(self.no)]        
        oo = self.softmax(netsOuput)        
        return oh,oo

    def backPropagate(self,train_row,oh,oo):
        """
        Melakukan backpropagation (menghitung error, update weight dan bias)
        return error(total) untuk semua neuron pada output layer         
        """
        inputs = train_row[0]
        targets = train_row[1]        
        
        #----------- 1. Hitung cost function (error) -------------------#        
        error_total = self.logLoss(targets,oo)

        new_inputs = [1]+inputs # [1] for bias

        #----------- 2. Update weight dan bias dari input ke hidden layer -------------------#    

        # for each neuron i on hidden layer
        for i in range(self.nh):
            #for each neuron j in inputs layer
            for j in range(self.ni+1):                
                delta_weight = self.deltaInputHidden(targets,oo,oh[i],new_inputs[j],i,self.who)                
                self.wih[i][j] = self.updateWeight(self.wih[i][j],delta_weight)                


        #----------- 3. Update weight dan bias dari hidden ke output layer -------------------#

        new_oh = [1]+oh # [1] for bias        
        #for each neuron i in output layer
        for i in range(self.no):            
            # for each neuron j in hidden layer
            for j in range(self.nh+1):                
                delta_weight = self.deltaHiddenOuput(targets[i],oo[i],new_oh[j])
                self.who[i][j] = self.updateWeight(self.who[i][j],delta_weight)                                    
        
        return error_total
    
    def train(self,train_row,validation_row=None): 
        """
        Melakukan training model
        """  

        print("Training.....")             
        targets = [row[1] for row in train_row]
        for i in range(self.EPOCH):
            errors = []
            predictions = []            
            for row in train_row:
                inputs = row[0]             
                oh,oo = self.forwardPass(inputs,self.wih,self.who)                
                predict = self.predict(oo)                          
                predictions.append(predict)
                error = self.backPropagate(row,oh,oo)
                errors.append(error)                              

            #if validation                                                  
            if validation_row is not None:
                self.validate(validation_row,self.wih,self.who)                
            
            #simpan error
            self.train_errors.append(np.mean(errors))

            #evaluate accuracy
            acc = self.getAccuracy(targets,predictions)
            self.train_accuracy.append(acc)
        
        print("Training completed.")

    def validate(self, validation_row,wih,who):
        """
        Melakukan validation pada data validation
        """        
        predictions=[]
        errors =[]
        target_set = []
        for row in validation_row:
            inputs = row[0]
            targets = row[1]
            target_set.append(targets)

            _,oo = self.forwardPass(inputs,wih,who)
            predict = self.predict(oo)
            predictions.append(predict)
            error = self.logLoss(targets,oo)              
            errors.append(error)                                            
        acc = self.getAccuracy(target_set,predictions)
        self.validation_accuracy.append(acc)
        self.validation_errors.append(np.mean(errors))        

    def squaredError(self,targets,outputs):
        sigma = 0
        for i in range(self.no):
            sigma += 1/2*(targets[i]-outputs[i])**2
        return sigma
    
    def logLoss(self,targets,outputs):        
        sigma = 0
        for i in range(self.no):
            sigma += targets[i]*math.log(outputs[i])                       

        return -sigma

    def deltaHiddenOuput(self,target,output_o,output_h):
        return (output_o-target)*output_h

    def deltaInputHidden(self,targets,outputs,output_h,input_i,h_index,who):
        sigma = 0
        for o in range(self.no):
            sigma += (outputs[o]-targets[o])*who[o][h_index]*(output_h*(1-output_h))*input_i
        return sigma

    def updateWeight(self,weight,delta_weight):
        return weight-self.LEARNING_RATE*delta_weight    
    
    def sigmoid(self,nets):        
        return [1/(1+math.e**(-net)) for net in nets]        

    def softmax(self,nets):
        summ=0
        for net in nets:
            summ+=math.exp(net)

        return [math.exp(net)/summ for net in nets]
        
    def netResult(self,inputs,weights):        
        net = weights[0]
        for i in range(len(inputs)):
            net+=inputs[i]*weights[i+1]
        
        return net

    def predict(self,oo):
        prediction=[0]*self.no
        predict_idx = np.argmax(oo)
        prediction[predict_idx]=1
        # return 0 if x<0.5 else 1        
        return prediction
                    
    def drawError(self,train=True,validation=True):
        if train:            
            plt.plot(self.train_errors,label="Train error")            
        if validation:
            plt.plot(self.validation_errors,label="Validation Error")
        plt.xlabel("Epoch")
        plt.ylabel("Log loss")
        plt.title("Grafik Error, learning rate = "+str(self.LEARNING_RATE))
        plt.legend()
        # plt.savefig("Grafik Error, learning rate = "+str(self.LEARNING_RATE)+".jpg")
        plt.show()
    
    def drawAccuracy(self,train=True,validation=True):
        if train:
            plt.plot(self.train_accuracy,label="Train accuracy")
        if validation:
            plt.plot(self.validation_accuracy,label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Grafik akurasi, learning rate = "+str(self.LEARNING_RATE))
        plt.legend()
        # plt.savefig("Grafik akurasi, learning rate = "+str(self.LEARNING_RATE)+".jpg")
        plt.show()

    def getAccuracy(self,targets,predictions):
        c=0
        for i in range(len(targets)):
            if targets[i]==predictions[i]:
                c+=1
        
        return c/len(targets)
        
# shuffle dataset order
def shuffle(dataset,balance=True):
    dataset2=dataset[:]
    if balance == True:
        for i in range(50):
            for j in range(3): 
                idx=j+3*i                       
                dataset2[idx]=dataset[j*50+i]                
    else:
        for i in range(len(dataset)):
            j = random.randint(0,i)
            dataset2[i],dataset2[j]=dataset2[j],dataset2[i]            
    
    return dataset2

def dataset_split(dataset,ratio=0.7):
    split_index = int(ratio*len(dataset))
    return dataset[:split_index],dataset[split_index:]

"""
### Dataset structure
### x adalah input dan t adalah target {[0,0,1],[0,1,0], atau [1,0,0]}

dataset =[
    [[x1, x2, x3, x4], [t1, t2, t3]],
    [[x1, x2, x3, x4], [t1, t2, t3]],
    [[x1, x2, x3, x4], [t1, t2, t3]],
    [[x1, x2, x3, x4], [t1, t2, t3]],
    [[.., .., .., ..], [.., .., ..]],
    [[.., .., .., ..], [.., .., ..]],
]
"""

### Read iris file

with open('D:\\Kuliah\\6 - Pembelajaran Mesin\\tugas4\\iris.csv') as f:
    reader = csv.reader(f)    
    iris = list(reader)

iris=list(filter(None,iris))

### making of dataset structure
classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
dataset = []
for row in iris:
    inputs = [float(x) for x in row[:-1]]
    targets = [0,0,0]
    targets[classes.index(row[-1])]=1
    dataset.append([inputs,targets])


dataset= shuffle(dataset,balance=True)
train_set,validation_set = dataset_split(dataset,0.7)

input_number = 4
hidden_number = 3
ouput_number = 3

#initialize weights
# wih = [[random.uniform(0.1,0.5) for i in range(input_number+1)] for j in range(hidden_number)]
# who = [[random.uniform(0.1,0.5) for i in range(hidden_number+1)] for j in range(ouput_number)]

#Neural network main
nn1 = NNetwork(input_number,hidden_number,ouput_number,0.8,500)
nn1.train(train_set,validation_set)
nn1.drawError()
nn1.drawAccuracy()

nn2 = NNetwork(input_number,hidden_number,ouput_number,0.1,500)
nn2.train(train_set,validation_set)
nn2.drawError()
nn2.drawAccuracy()



