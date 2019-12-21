import tensorflow as tf
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
import os, time, argparse
from sklearn.cluster import SpectralClustering
import sklearn.metrics as sklm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataloader import Dataloader
from nets import GALA


home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

parser.add_argument("--train_dir", default="test", type=str)
parser.add_argument("--dataset", default="mnist", type=str)
args = parser.parse_args()

def get_affinity(path, data, num_cluster = 20, ):
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(data, num_cluster, mode='connectivity', include_self=True)
    A.toarray()
    A = A + A.T
    A = (A != 0)*np.exp(-1)
    ss.save_npz(path + '/pre_built/test_full.npz', A)
    return A

if __name__ == '__main__':
    ### define path and hyper-parameter
    gpu_num = 0
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')

    _,_,test_images, test_labels, pre_processing = Dataloader(args.dataset, '')

    test_images = test_images.reshape(test_images.shape[0],-1).astype(np.float32)
    test_images = test_images/255
    num_labels = np.max(test_labels)+1

    if os.path.isfile( home_path+'/pre_built/test_full.npz' ):   
        A = ss.load_npz(home_path+'/pre_built/test_full.npz')
    else:
        A = get_affinity(home_path, test_images)
    A_ = ss.csr_matrix(A.reshape(-1,1))
    indices = np.vstack([A_.indices//A.shape[0],A_.indices%A.shape[0]]).T
    values = np.exp(-1)*np.ones([indices.shape[0]], dtype = np.float32)
    A = {'indices' : indices, 'values' : values, 'dense_shape' : A.shape}

    model = GALA.Model(weight_decay = 0., A = A, 
                       name = 'GALA', trainable = False)
    model.update_DAD()
    H = model(test_images, training = False)

    trained = sio.loadmat(args.train_dir + '/trained_params.mat')
    n = 0
    for v in model.non_trainable_variables:
        v.assign(trained[v.name].reshape(*v.shape))
        n += 1
    print (n, 'params loaded')
    t = time.time()
    H = model(test_images, training = False)
    latent = H.numpy()

    l = 5e1
    mu = 1.
    
    u,s,v = np.linalg.svd(mu*np.eye(latent.shape[1]) + l*np.matmul(latent.T, latent), full_matrices = False)
    H_inv = np.matmul(v.T, (u/np.expand_dims(s,0)).T)
    A_latent = l*np.matmul(np.matmul(latent, H_inv), latent.T)
    A_latent = (A_latent - np.min(A_latent))/(np.max(A_latent)+np.min(A_latent))
    print ('Optimal A is computes')

    clustering = SpectralClustering(n_clusters=num_labels, affinity = 'precomputed')
    prediction = clustering.fit(A_latent)

    results = prediction.labels_

    nmi_score = sklm.adjusted_mutual_info_score(test_labels.reshape(-1), results.reshape(-1))
    ari_score = sklm.adjusted_rand_score(test_labels.reshape(-1), results.reshape(-1))
    print('NMI : %.4f, ARI : %.4f'%(nmi_score, ari_score))
    print('test_time : %.4f'%(time.time()-t))

