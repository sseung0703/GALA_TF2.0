import tensorflow as tf
import numpy as np
from sklearn.cluster import SpectralClustering
import scipy.sparse as ss
import sklearn.metrics as sklm
import cv2
from nets import SVD
def Optimizer(model, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        optimizer = tf.keras.optimizers.Adam(LR[0])
        optimizer_tune = tf.keras.optimizers.Adam(LR[1])
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        ACC = tf.keras.metrics.Mean(name='ACC')
        NMI = tf.keras.metrics.Mean(name='NMI')
        ARI = tf.keras.metrics.Mean(name='ARI')
        
    l = 5e1
    mu = 1.
    @tf.function
    def training(input, labels, weight_decay, k):
        with tf.GradientTape() as tape:
            generated = model(input, training = True)
            loss = tf.reduce_sum(tf.square(input - generated))/2/input.shape[0]
            total_loss = loss
            if weight_decay > 0.:
                total_loss += tf.add_n([tf.reduce_sum(tf.square(v))*weight_decay/2 for v in model.trainable_variables])
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        
    @tf.function
    def finetuning(input, labels, weight_decay, k):
        with tf.GradientTape() as tape:
            generated = model(input, training = True)
            loss = tf.reduce_sum(tf.square(input - generated))/2/input.shape[0]
            
            H = model.H
            HHT = tf.matmul(H,H,transpose_a=True)
            s,u,v = SVD.SVD(tf.expand_dims(mu*tf.eye(HHT.shape[0])+l*HHT,0), k)
            scc_loss = mu*l/2*tf.linalg.trace(tf.matmul(tf.squeeze(tf.matmul(v, u/tf.reshape(s,[1,1,k]), transpose_b = True)),HHT))
            total_loss = loss + scc_loss
            
            if weight_decay > 0.:
                total_loss += tf.add_n([tf.reduce_sum(tf.square(v))*weight_decay for v in model.trainable_variables])
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer_tune.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        
    def validate(input, labels, k):
        H = model(input, training = False)
        H = H.numpy()
        cv2.imwrite('test2/rec0.png',(np.clip(np.hstack(list(np.hstack(list(H[:400].reshape(20,20,28,28))))),0,1)*255).astype(np.uint8))
        cv2.imwrite('test2/rec1.png',(np.clip(np.hstack(list(np.hstack(list(H[-400:].reshape(20,20,28,28))))),0,1)*255).astype(np.uint8))
        cv2.imwrite('test2/ori.png',(np.clip(np.hstack(list(np.hstack(list(input[:400].reshape(20,20,28,28))))),0,1)*255).astype(np.uint8))
        H = model.H
        latent = H.numpy()
    
        u,s,v = np.linalg.svd(mu*np.eye(latent.shape[1]) + l*np.matmul(latent.T, latent), full_matrices = False)
        H_inv = np.matmul(v[:k].T, (u[:,:k]/np.expand_dims(s[:k],0)).T)
        A_latent = l*np.matmul(np.matmul(latent, H_inv), latent.T)
        A_latent = np.maximum(0,A_latent)
        #ss.save_npz('test2/test_full.npz', ss.csr_matrix(A_latent))
        print ('Optimal A is computes')
        
        num_labels = np.max(labels)+1
    
        clustering = SpectralClustering(n_clusters = num_labels, affinity = 'precomputed')
        prediction = clustering.fit(A_latent)
    
        results = prediction.labels_
        
        total_true = 0
        vote_box = {c:0 for c in range(num_labels)}
        for i in range(num_labels):
            matched = labels[results == i]
            for m in matched:
                if vote_box.get(m) is not None:
                    vote_box[m] += 1
                   
            num_true = 0 
            cluster_id = 0
            for v in vote_box.keys():
                if vote_box[v] > num_true:
                    cluster_id = v
                    num_true = vote_box[v]
                    
            total_true += num_true
            
            if num_true != 0:
                del vote_box[cluster_id]
            for v in vote_box.keys():
                vote_box[v] = 0
            
        acc = total_true/labels.shape[0]
    
        nmi_score = sklm.adjusted_mutual_info_score(labels.reshape(-1), results.reshape(-1))
        ari_score = sklm.adjusted_rand_score(labels.reshape(-1), results.reshape(-1))
        ACC.update_state(acc)
        NMI.update_state(nmi_score)
        ARI.update_state(ari_score)
        
    return training, train_loss, finetuning, validate, ACC, NMI, ARI
    
    
    
    
    
    
    
    