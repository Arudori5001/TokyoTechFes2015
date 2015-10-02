#coding: utf-8
'''
Created on 2015/09/16

@author: kakaminaoto
'''

class Record:
    '''
    @summary: 機械学習における教師データを表します。
    '''


    def __init__(self, input, label):
        '''
        @param input: np.array モデルへの入力
        label: int 教師データのラベル
        '''
        self.input = input
        self.label = label
        