import tensorflow as tf
class Model(tf.keras.Model):
    def __init__(self, DAD, name = 'GALA', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        
        self.GALA = {}
        D = [800,700,500]
        for i, d in enumerate(D):
            self.GALA['enc%d'%i] = tf.keras.layers.Dense(d, trainable = trainable)#, use_bias = False)
        if trainable:
            for i, d in enumerate(D[1::-1]+[28**2]):
                self.GALA['dec%d'%i] = tf.keras.layers.Dense(d, trainable = trainable)#, use_bias = False)
        
        self.DADsm = tf.sparse.SparseTensor(DAD['DADsm_indices'], DAD['DADsm_values'][0], DAD['dense_shape'][0])
        self.DADsp = tf.sparse.SparseTensor(DAD['DADsp_indices'], DAD['DADsp_values'][0], DAD['dense_shape'][0])
        
    def Laplacian_smoothing(self, x, name, training):
        return tf.nn.relu(tf.sparse.sparse_dense_matmul(self.DADsm, self.GALA[name](x, training = training)))
    
    def Laplacian_sharpening(self, x, name, training):
        return tf.nn.relu(tf.sparse.sparse_dense_matmul(self.DADsp, self.GALA[name](x, training = training)))
        
    def call(self, H, training=None):
        for i in range(3):
            H = self.Laplacian_smoothing(H, 'enc%d'%i, training)
        self.H = H
        #if training == False:
            #return H
        for i in range(3):
            H = self.Laplacian_sharpening(H, 'dec%d'%i, training)
        return H