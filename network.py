# -*- coding:utf-8 -*-
import numpy as np
import random
import time
import load

class Network(object):# extending a class called object. It's not an argument.
    def __init__(self,sizes):
        self.sizes=sizes #sizes contains the number of neurons in the respective layers
        self.num_lyers=len(sizes)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]] #np.random.randn function to generate Gaussian distributions with mean ￼ and standard deviation ￼
        # self.weights=[np.random.randn(x,y)for x,y in zip(sizes[:-1],sizes[1:])]
        # self.weights=[np.random.randn(x,y) for y ,x in zip(sizes[:-1],sizes[1:])]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

     #retrun the output activations
    def feedforward(self , a): # a 是神经网络的第一层训练数据输入参数
        for w,b in zip(self.weights,self.biases):
            a=sigmoid_vec(np.dot(w,a)+b)
        return a

    def SGD(self,train_set,epochs,mini_batch_size,eta,test_data=None):
        """
            neural network uses stochastic gradient descent:mini-batch stochastic
        gradient descent.
        `eta``is the learning rate.
        """
        time_begin=time.localtime()
        print "Neural Network begin at {0}".format(time.strftime("%Y-%m-%d %H:%M:%S"),time_begin)
        print "The parameters of current Neural Network is : " \
              "Network size is:{2} ,mini batch size is {0}, learning rate is:{1}  ".format(mini_batch_size,eta,self.sizes)
        if test_data:
            test_number=len(test_data)
        train_number=len(train_set)
        for j in xrange(epochs):
            random.shuffle(train_set) # 将数据打乱保证数据随机分布
            mini_batchs=[train_set[k:mini_batch_size+k] for k in xrange(0,train_number,mini_batch_size )]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta,j)
            if test_data:
                test_correct_number=self.evaluate(test_data)
                train_correct_number=self.evaluate_train(train_set)
                test_precision=float(test_correct_number)/test_number
                train_precision=float(train_correct_number)/train_number

                print "After epoch {0} the result is : {1}/{2} ;train precision is: {3}".format(j+1,train_correct_number,train_number,train_precision)
                print "After epoch {0} the result is : {1}/{2} ;test precision is : {3}".format(j+1,test_correct_number,test_number,test_precision)
            else:
                print "Epoch {}end !".format(j)
        time_end=time.localtime()
        print "Neural Network end  at {0}".format(time.strftime("%Y-%m-%d %H:%M:%S"),time_end)
        print "Runtime of Neural Network with {0} epochs is : {1}秒".format(epochs,time.mktime(time_end)-time.mktime(time_begin))





    def update_mini_batch(self,mini_batch,eta,epoch):
        """
        :param mini_batch:
        :param eta:
        :return none
        """
        babla_b=[np.zeros(b.shape) for b in self.biases] #  sum of  all in mini_batch dataset
        babla_w=[np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_babla_b,delta_babla_w=self.backpro(x,y)

            """
            check with checking_SGD_w function ,if partial dericative is correct
            """
            # if epoch>0 :
            #     delta_babla_wlij=delta_babla_w[1][10][5]
            #     check_delta_babla_wlij=self.checking_SGD_w(x,0.0001,1,10,5)
            #     print "check number :{0:.15f}".format(check_delta_babla_wlij)
            #     print "SGD number   :{0:.15f}".format(delta_babla_wlij)



            babla_w=[dbw+bw for dbw ,bw in zip(delta_babla_w,babla_w)]
            babla_b=[dbb+bb for dbb,bb in zip(delta_babla_b,babla_b)]

        self.weights=[w-eta*bw/len(mini_batch) for w ,bw in zip(self.weights,babla_w)]
        self.biases=[b-eta*bb/len(mini_batch) for b,bb in zip(self.biases,babla_b) ]






    def backpro(self,x,y):
        babla_b=[np.zeros(b.shape) for b in self.biases] # for a dataset
        babla_w=[np.zeros(w.shape) for w in self.weights]

        activation=x
        activations=[x]     # all a=f(z) :f=sigmoid in this neural network
        zs=[]               #all z=w*a+b layer by layer
        i=0
        for b,w in zip(self.biases,self.weights): # feed forward process

            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid_vec(z)
            activations.append(activation)


        # back propagation  process

        delta=self.cost_derivative(activations[-1],y) * sigmoid_prime_vec(z)
        babla_b[-1]=delta
        babla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in xrange(2,self.num_lyers):
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime_vec(zs[-l])
            babla_b[-l]=delta
            babla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (babla_b,babla_w)


    def evaluate(self,test_date): #calculate total number of true predict in test dataset
        # correct_number=0
        # for x,y in test_date:
        #    if  np.argmax(self.feedforward(x))==y:
        #        correct_number+=1

        results=[ (np.argmax(self.feedforward(x)),y)  for x,y in test_date] # np.arumax is to get the index of the lagest number in np.ndarray
        correct_number=sum([int(y_predict==y) for y_predict,y in results])
        return  correct_number

    def evaluate_train(self,train_data): #calculate total number of true predict in test dataset
        # correct_number=0
        # for x,y in test_date:
        #    if  np.argmax(self.feedforward(x))==y:
        #        correct_number+=1

        results=[ (np.argmax(self.feedforward(x)),y)  for x,y in train_data] # np.arumax is to get the index of the lagest number in np.ndarray
        correct_number=sum([int(y_predict==np.argmax(y)) for y_predict,y in results])
        return  correct_number

    def cost_derivative(self,output_activations,y): #(yi -ai)
          # y  在训练数据集中（10 1）的向量，但是注意在测试集中为一个数值
        return output_activations-y

    def checking_SGD_w(self,x,EPSILON,l,i,j):

        """

        :param EPSILON:
         i j l denotes the w of l layers ,index is (i,j)
        :return limit  partial derivative of  w of l layer with index of (i,j)
        这里要注意防止出现浅拷贝
        例如： check_babla_w_plus=[w for w in self.weights]
        这里的 check_babla_w_plus 和weights是同一片内存区域，同时发生改变，只是名称不同

        """
        check_babla_w_plus=[np.zeros(w.shape)+w for w in self.weights]
        check_babla_w_sub=[np.zeros(w.shape)+w for w in self.weights]

        check_babla_w_plus[l][i][j]+=EPSILON
        check_babla_w_sub[l][i][j]=check_babla_w_sub[l][i][j]-EPSILON
        activation1=x
        activation2=x
        for w,b in zip(check_babla_w_plus,self.biases):
            activation1=sigmoid_vec(np.dot(w,activation1)+b)
        for w,b in zip(check_babla_w_sub,self.biases):
            activation2=sigmoid_vec(np.dot(w,activation2)+b)
        # print activation1
        # print activation2
        check_partial_derivative=(activation1[np.argmax(activation1)][0]-activation2[np.argmax(activation2)][0])/(2*EPSILON)
        return  check_partial_derivative

#define sigmoid functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)


# define derivative sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def tangent(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


"""
test network
"""

import load
training_data, validation_data, test_data =load.load_data_wrapper()

net = Network([784, 30,10])
net.SGD(training_data, 20, 10, 0.1, test_data=test_data)




