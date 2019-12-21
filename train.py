import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import os, time, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataloader import Dataloader
import op_util
from nets import GALA

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="test", type=str)
parser.add_argument("--dataset", default="mnist", type=str)
args = parser.parse_args()

def get_affinity(path, data, num_cluster = 20, ):
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(data, num_cluster, mode='connectivity', include_self=False)
    A.toarray()
    A = A*np.exp(-1)
    A = A + A.T
    A = ss.csr_matrix.todense(A)
    eye = np.eye(A.shape[0])
    
    Asm = A + eye
    Dsm = 1/np.sqrt(np.sum(Asm, -1))
    DADsm = ss.csr_matrix(np.multiply(np.multiply(Asm, Dsm).T, Dsm).reshape(-1))
    
    Asp = 2*eye - A
    Dsp = 1/np.sqrt(np.sum(2*eye + A, -1))
    DADsp = ss.csr_matrix(np.multiply(np.multiply(Asp, Dsp).T, Dsp).reshape(-1))
    
    DADsm_indices = np.vstack([DADsm.indices//A.shape[0],DADsm.indices%A.shape[0]]).T
    DADsp_indices = np.vstack([DADsp.indices//A.shape[0],DADsp.indices%A.shape[0]]).T
    DAD = {'DADsm_indices' : DADsm_indices, 'DADsm_values' : DADsm.data.astype(np.float32), 
           'DADsp_indices' : DADsp_indices, 'DADsp_values' : DADsp.data.astype(np.float32), 'dense_shape' : A.shape}
    sio.savemat(path + '/pre_built/test_full.mat', DAD)
    return DAD

if __name__ == '__main__':
    ### define path and hyper-parameter
    train_lr = 1e-4
    finetune_lr = 1e-6
    weight_decay = 5e-4
    k = 20
    maximum_epoch = 10000
    early_stopping = 20
    finetune_epoch = 50
    
    do_log  = 100
    do_test = 1000
    
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
   
    _,_, test_images, test_labels = Dataloader(args.dataset, '')
    test_images = test_images.reshape(test_images.shape[0],-1).astype(np.float32)
    test_images = test_images/255
    
    
    if os.path.isfile( home_path+'/pre_built/test_full.mat' ):
        DAD = sio.loadmat(home_path+'/pre_built/test_full.mat')
    else:
        DAD = get_affinity(home_path, test_images, k)
    model = GALA.Model(DAD = DAD, name = 'GALA', trainable = True)
    
    init_step, init_loss, finetuning, validate, ACC, NMI, ARI = op_util.Optimizer(model, [train_lr, finetune_lr])
   
    summary_writer = tf.summary.create_file_writer(args.train_dir)
    with summary_writer.as_default():
        step = 0
        
        best_loss = 1e12
        stopping_step = 0
        
        train_time = time.time()
        for epoch in range(maximum_epoch):
#            init_step(test_images, test_labels)
            init_step(test_images, test_labels, weight_decay, k)
            step += 1
            
            if epoch%do_log == 0 or epoch == maximum_epoch-1:
                template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                print (template.format(step, init_loss.result(), (time.time()-train_time)/do_log))
                train_time = time.time()
                current_loss = init_loss.result()
                tf.summary.scalar('Initialization/train',current_loss , step=epoch+1)
                init_loss.reset_states()
                
            if epoch%do_test == 0 or epoch == maximum_epoch-1:
                validate(test_images, test_labels, k)
                tf.summary.scalar('Metrics/ACC', ACC.result(), step=epoch+1)
                tf.summary.scalar('Metrics/NMI', NMI.result(), step=epoch+1)
                tf.summary.scalar('Metrics/ARI', ARI.result(), step=epoch+1)
                    
                template = 'Epoch: {0:3d}, NMI: {1:0.4f}, ARI.: {2:0.4f}'
                print (template.format(epoch+1, NMI.result(), ARI.result()) )
                
                NMI.reset_states()
                ARI.reset_states()

                params = {}
                for v in model.variables:
                    params[v.name] = v.numpy()
                sio.savemat(args.train_dir+'/trained_params.mat', params)
            
        
        if finetune_epoch > 0:
            train_time = time.time()
            for epoch in range(finetune_epoch):
                finetuning(test_images, test_labels, weight_decay, k)
                step += 1
                
            validate(test_images, test_labels, k)
            tf.summary.scalar('Metrics/ACC', ACC.result(), step=maximum_epoch+epoch+1)
            tf.summary.scalar('Metrics/NMI', NMI.result(), step=maximum_epoch+epoch+1)
            tf.summary.scalar('Metrics/ARI', ARI.result(), step=maximum_epoch+epoch+1)
                
            template = 'Epoch: {0:3d}, ACC: {1:0.4f}, NMI: {2:0.4f}, ARI.: {3:0.4f}'
            print (template.format(epoch+1, ACC.result(), NMI.result(), ARI.result()) )
            
            NMI.reset_states()
            ARI.reset_states()
        
        params = {}
        for v in model.variables:
            params[v.name] = v.numpy()
        sio.savemat(args.train_dir+'/tuned_params.mat', params)
            
                
            
            
