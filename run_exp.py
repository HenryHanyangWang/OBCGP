import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from botorch.test_functions import Beale,Branin,SixHumpCamel,Hartmann
from botorch.utils.transforms import normalize

from OBCGP.utilis import obj_fun, get_initial_points
from OBCGP.GP import ELBO,acq_fcn_EI,par_update
import obj_functions.push_problems
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []

# temp={}
# temp['name']='Push4D' 
# f_class = obj_functions.push_problems.push4
# tx_1 = 3.5; ty_1 = 4
# fun = f_class(tx_1, ty_1)
# temp['function'] = fun
# temp['fstar'] =  0.
# function_information.append(temp)

temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=True)
temp['fstar'] =  -0.397887 
function_information.append(temp)

# temp={}
# temp['name']='SixHumpCamel2D' 
# temp['function'] = SixHumpCamel(negate=True)
# temp['fstar'] =  1.0317
# function_information.append(temp)

# temp={}
# temp['name']='Hartmann3D' 
# temp['function'] = Hartmann(dim=3,negate=True)
# temp['fstar'] =  3.86278
# function_information.append(temp)



for information in function_information:

    fun = information['function']
    fstar = information['fstar']
    # print('fstar is: ',fstar)
   
    neg_fun = obj_fun(fun)

    dim=neg_fun.dim
    bounds=neg_fun.bounds
    n_init =4*dim
    priori_known_mode=True
    eps = 1e-4
    Max_iter = 20
    lr = 1e-1


    upp_bdd_prior = fstar


    if dim <=5:
        step_size = 3
        iter_num = 100
        N = 2
    elif dim<=8:
        step_size = 3
        iter_num = 150
        N = 100
    else:
        step_size = 3
        iter_num = 200
        N = 100

    OBCGP = []

    for exp in range(N):

        X = get_initial_points(bounds=bounds,num=n_init,device=device,dtype=dtype,seed=exp)
        X = normalize(X, bounds)
        X = X.numpy()
        Y=neg_fun(X)
        Y_best=np.max(Y)
        Y_rescale=Y-Y_best

        low_bdd_feed=0.0
        upp_bdd_feed=upp_bdd_prior-Y_best

        lamda_feed = np.float32(np.log(10.0))
        alpha_feed = np.float32(0.0)
        beta_feed = lamda_feed
        X_M_t = np.reshape(X[np.argmax(Y)], (1, dim))
        X_M_t_feed = np.log(np.divide(X_M_t, 1.0 - X_M_t + eps)) + np.random.normal(0.0, 0.01, [1, dim])

        best_record = []
        best_record.append(-np.max(Y))
        print('initial best is: ',best_record[-1])

        for iter in range(100):

            tf.set_random_seed(1234)
            np.random.seed(1234) 

            try:
                if iter%step_size == 0:
                    sigma_feed, phi_feed = par_update(X, Y_rescale,dim)

                tf.reset_default_graph()
                alpha_v = tf.Variable(tf.convert_to_tensor(alpha_feed, dtype=tf.float32))
                beta_v = tf.Variable(tf.convert_to_tensor(beta_feed, dtype=tf.float32))
                X_M_t_v = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
                X_M = (tf.sigmoid(X_M_t_v))
                x_data_p = tf.convert_to_tensor(X, dtype=tf.float32)
                y_data_p = tf.convert_to_tensor(Y_rescale, dtype=tf.float32)
                low_bdd_p = tf.convert_to_tensor(low_bdd_feed, tf.float32)
                upp_bdd_p=tf.convert_to_tensor(upp_bdd_feed,tf.float32)
                lamda_p = tf.convert_to_tensor(lamda_feed, dtype=tf.float32)
                phi_p = (tf.convert_to_tensor(phi_feed, dtype=tf.float32))
                sigma_p = (tf.convert_to_tensor(sigma_feed, dtype=tf.float32))

                pt_sel = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
                costF = ELBO(x_data_p, y_data_p, low_bdd_p,upp_bdd_p, X_M, phi_p, sigma_p, alpha_v, beta_v, lamda_p)
                q_mu_p=tf.exp(alpha_v) / (tf.exp(alpha_v) + tf.exp(beta_v))
                q_var_p=tf.divide(tf.exp(alpha_v+beta_v),tf.square(tf.exp(alpha_v)+tf.exp(beta_v))*(1.0+tf.exp(alpha_v)+tf.exp(beta_v)))
                acq_fcn_val = acq_fcn_EI(tf.sigmoid(pt_sel), x_data_p, y_data_p, low_bdd_p,upp_bdd_p, X_M, phi_p, sigma_p,q_mu_p,q_var_p)

                optimizer = tf.train.AdamOptimizer(lr)
            
                train_par = optimizer.minimize(costF)
                train_acq = optimizer.minimize(-acq_fcn_val, var_list=pt_sel)

                sess = tf.Session()
                init = tf.global_variables_initializer()
                sess.run(init)

                for opt_iter in range(Max_iter):
                    sess.run(train_par)
                for opt_iter_acq in range(Max_iter):
                    sess.run(train_acq)
                X_M_np, phi_np, sigma_np, alpha_np, beta_np = sess.run((X_M, phi_p, sigma_p, alpha_v, beta_v))
                q_mu = np.exp(alpha_np - beta_np)
                q_var = np.exp(alpha_np - 2.0 * beta_np)

                pt_next=sess.run(tf.sigmoid(pt_sel))
                val_next = np.reshape(neg_fun(pt_next), [])

                X = np.concatenate((X, pt_next), axis=0)
                Y = np.concatenate((Y, np.reshape(val_next, [1, 1])), axis=0)

                if val_next > Y_best:
                    X_M_t_feed = np.log(np.divide(pt_next + eps, 1.0 - pt_next + eps)) + np.random.normal(0.0, 0.01, [1, dim])
                    alpha_feed = np.float32(0.0)
                    beta_feed = lamda_feed
                    X_M_t = pt_next
                    Y_best = val_next
                    low_bdd_feed = 0.0
                    upp_bdd_feed= upp_bdd_prior-val_next
                    Y_rescale = Y - Y_best
                else:
                    X_M_t_feed = np.log(np.divide(X_M_t + eps, 1.0 - X_M_t + eps)) + np.random.normal(0.0, 0.01, [1, dim])
                    alpha_feed = np.float32(0.0)
                    beta_feed = lamda_feed
                    Y_rescale = Y - Y_best

                sess.close()
                tf.reset_default_graph()
                X_M_best = np.reshape(X[np.argmax(Y)], (1, dim))

                best_record.append( -np.max(Y))
            except:
                print('oh no!!')
                best_record.append( -np.max(Y))
            
            print('iteration #:%d\t' % (iter + 1) + 'current Minimum value: %f\t' % (-np.max(Y)) )

        OBCGP.append(best_record)

    np.savetxt('result/'+information['name']+'_OBCGP', OBCGP, delimiter=',')