import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from collections import deque


class ChainerModel:
    """Chainerによる学習モデル"""


    def __init__(self, model, optimizer,forward_function):
        self.model = model
        self.optimizer = optimizer
        self.forward = forward_function


    def learn(self, x_train, y_train, x_test, y_test,
              max_epoch, look_back, prop_increase, batchsize, gpu_id,
              allow_dropout, dropout_ratio): 
        """学習を行う"""
    
        n_train = len(x_train)
        n_test = len(x_test)

        train_accs = []
        test_accs = []
        deq = deque()
        epoch = 1
        while self.decide_continue(deq, look_back, prop_increase) and epoch <= max_epoch:
            print('epoch ' + str(epoch))
    
            # training
            perm = np.random.permutation(n_train)
            sum_accuracy = 0
            sum_loss = 0
            for i in range(0, n_train, batchsize):
                x_batch = x_train[perm[i:i+batchsize]]
                y_batch = y_train[perm[i:i+batchsize]]
                if gpu_id >= 0:
                    x_batch = cuda.to_gpu(x_batch)
                    y_batch = cuda.to_gpu(y_batch)

                self.optimizer.zero_grads()
                y = self.forward(self.model, x_batch, ratio=dropout_ratio, train=allow_dropout)
                t = chainer.Variable(y_batch)
                loss = F.softmax_cross_entropy(y,t)
                acc = F.accuracy(y,t)
                loss.backward()
                self.optimizer.update()

                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
            
            train_acc = sum_accuracy / n_train
            print('train mean loss={}, accuracy={}'.format(
                sum_loss / n_train, train_acc))
            train_accs.append(train_acc)
    
            # evaluation
            sum_accuracy = 0
            sum_loss     = 0
            for i in range(0, n_test, batchsize):
                x_batch = x_test[i:i+batchsize]
                y_batch = y_test[i:i+batchsize]
                if gpu_id >= 0:
                    x_batch = cuda.to_gpu(x_batch)
                    y_batch = cuda.to_gpu(y_batch)
    
                y = self.forward(self.model, x_batch, ratio=dropout_ratio, train=allow_dropout)
                t = chainer.Variable(y_batch)
                loss = F.softmax_cross_entropy(y,t)
                acc = F.accuracy(y,t)
    
                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
                
            test_acc = sum_accuracy / n_test
            print('test  mean loss={}, accuracy={}'.format(
                sum_loss / n_test, test_acc))
            test_accs.append(test_acc)
            deq.appendleft(test_acc)

            epoch += 1
    
        return train_accs,test_accs
        

    def decide_continue(self, deq, look_back, prop_increase):
        if len(deq) < look_back:
            return True
        elif (deq[0] - deq.pop()) < prop_increase:
            return False
        else:
            return True



    


