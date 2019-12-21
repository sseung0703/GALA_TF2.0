import tensorflow as tf

def removenan(x):
    return tf.where(tf.math.is_finite(x), x,tf.zeros_like(x))

def SVD(X, n, method = 'SVD'):
    sz = X.get_shape().as_list()
    if len(sz)==4:
        x = tf.reshape(X,[-1,sz[1]*sz[2],sz[3]])
    elif len(sz)==3:
        x = X
    else:
        x = tf.expand_dims(X, 1)
        n = 1
    _, HW, D = x.get_shape().as_list()

    s,u,v = SVD_custom(x, n)

    return s, u, v

@tf.custom_gradient
def SVD_custom(x, n):
    with tf.device('CPU'):
        s, U, V =  tf.linalg.svd(x)
    s = removenan(s)
    V = removenan(V)
    U = removenan(U)
    s = tf.nn.l2_normalize(tf.slice(s,[0,0],[-1,n]),1)
    U = tf.nn.l2_normalize(tf.slice(U,[0,0,0],[-1,-1,n]),1)
    V = tf.nn.l2_normalize(tf.slice(V,[0,0,0],[-1,-1,n]),1)
    
    def gradient_svd(ds, dU, dV):
        u_sz = dU.shape[1]
        v_sz = dV.shape[1]
        s_sz = ds.shape[1]

        S = tf.linalg.diag(s)
        s_2 = tf.square(s)

        eye = tf.expand_dims(tf.eye(s_sz),0) 
        k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
        KT = tf.transpose(k,[0,2,1])
        KT = removenan(KT)
    
        def msym(X):
            return (X+tf.transpose(X,[0,2,1]))
    
        def left_grad(U,S,V,dU,dV):
            U, V = (V, U); dU, dV = (dV, dU)
            D = tf.matmul(dU,tf.linalg.diag(1/(s+1e-8)))
    
            grad = tf.matmul(D - tf.matmul(U, tf.linalg.diag(tf.linalg.diag_part(tf.matmul(U,D,transpose_a=True)))
                               + 2*tf.matmul(S, msym(KT*(tf.matmul(D,tf.matmul(U,S),transpose_a=True))))), V,transpose_b=True)
        
            grad = tf.transpose(grad, [0,2,1])
            return grad

        def right_grad(U,S,V,dU,dV):
            grad = tf.matmul(2*tf.matmul(U, tf.matmul(S, msym(KT*(tf.matmul(V,dV,transpose_a=True)))) ),V,transpose_b=True)
            return grad
    
        grad = tf.cond(tf.greater(v_sz, u_sz), lambda :  left_grad(U,S,V,dU,dV), 
                                               lambda : right_grad(U,S,V,dU,dV))
        return [grad, None]
    return [s,U,V], gradient_svd
