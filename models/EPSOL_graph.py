from keras.layers import Embedding,Concatenate, Add
from keras.layers.convolutional import Conv1D
from keras.layers.core import *
from keras.models import *
from keras.layers.pooling import MaxPooling1D,GlobalMaxPooling1D
import keras

class EPSOL():
    def __init__(self):
        self.maxlen = 1200
        self.seq_size = 21
        self.embedding_size = 64
        self.em_drop = 0.2
        self.num_classes = 2
        self.num_bio = 57
        self.Kernel_size1 = [(3,32),(5,32),(7,32),(9,32),(11,32),(13,32),(15,32)]
        self.Kernel_size2 = [(3,32),(5,32),(7,32),(9,32),(11,64),(13,64),(15,64)]
        self.Kernel_size3 = [(2,32),(3,32),(4,32),(5,32),(6,32),(7,32),(8,32),(9,32),(10,32),(11,32),(12,32),(13,32),(14,32),(15,32)]
    
    def back_CNN(self,Kernel_size,embedding_dim):
        # x [,max_len,embedding_size]
        x = Input(shape=(self.maxlen,embedding_dim))
        conv_blocks = []
        for K in Kernel_size:
            conv = Conv1D(filters = K[1],
                         kernel_size=K[0],
                         padding='valid',
                         activation='relu',
                         strides=1,
                         kernel_initializer = 'normal')(x)
            conv = MaxPooling1D(pool_size=int(conv.shape[1]))(conv)
            conv = Flatten()(conv)
            # conv = Dropout(0.2)(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        back_CNN = Model(x,z)
        # print(back_CNN.summary())
        return back_CNN

    def get_model(self):
        input_1 = Input(shape=(self.maxlen,)) # seq input
        input_2 = Input(shape=(self.maxlen,)) # ngram2 input
        input_3 = Input(shape=(self.maxlen,)) # ngram3 input
        input_4 = Input(shape=(self.maxlen,)) # acc input
        input_5 = Input(shape=(self.maxlen,)) # acc20 input
        input_6 = Input(shape=(self.maxlen,)) # ss input
        input_7 = Input(shape=(self.maxlen,)) # ss8 input
        input_8 = Input(shape=(self.num_bio,)) # bio input

        # 64 64 64 5 21 5 10 0.7806
        # 64 64 64 5 32 5 10 0.7816 0.7826
        embedding_layer1 = Embedding(21,64,input_length=self.maxlen)
        embedding_layer2 = Embedding(401,64,input_length=self.maxlen)
        embedding_layer3 = Embedding(8001,64,input_length=self.maxlen)
        embedding_layer4 = Embedding(3,5,input_length=self.maxlen)
        embedding_layer5 = Embedding(21,32,input_length=self.maxlen)
        embedding_layer6 = Embedding(4,5,input_length=self.maxlen)
        embedding_layer7 = Embedding(9,10,input_length=self.maxlen)

        em1 = embedding_layer1(input_1)
        em2 = embedding_layer2(input_2)
        em3 = embedding_layer3(input_3)
        em4 = embedding_layer4(input_4)
        em5 = embedding_layer5(input_5)
        em6 = embedding_layer6(input_6)
        em7 = embedding_layer7(input_7)

        em1 = SpatialDropout1D(self.em_drop)(em1)
        em2 = SpatialDropout1D(self.em_drop)(em2)
        em3 = SpatialDropout1D(self.em_drop)(em3)
        em4 = SpatialDropout1D(self.em_drop)(em4)
        em5 = SpatialDropout1D(self.em_drop)(em5)
        em6 = SpatialDropout1D(self.em_drop)(em6)
        em7 = SpatialDropout1D(self.em_drop)(em7)

        z1 = self.back_CNN(self.Kernel_size1,64)(em1)
        z2 = self.back_CNN(self.Kernel_size1,64)(em2)
        z3 = self.back_CNN(self.Kernel_size1,64)(em3)
        z4 = self.back_CNN(self.Kernel_size1,5)(em4)
        z5 = self.back_CNN(self.Kernel_size1,32)(em5)
        z6 = self.back_CNN(self.Kernel_size1,5)(em6)
        z7 = self.back_CNN(self.Kernel_size1,10)(em7)
        
        z = Concatenate()([z1,z2,z3,z4,z5,z6,z7,input_8])
        # z = Concatenate()([z1,z2,z3,input_4])

        z = Dense(64,activation="relu")(z)
        z = Dropout(0.2)(z)

        main_output = Dense(self.num_classes, activation='softmax')(z)
        model = Model(inputs=[input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8], outputs = main_output)
        return model
