import numpy as np

#Calcolo il lr ad ofni iterazione usando il polinomial decay (decadimento polinomiale)
# Optimizer è l'ottimizzatore, init_lr è il learning rate iniziale, iter è l'iterazione corrente, lr_decay_iter è la frequenza di decadimento, max_iter è il numero massimo di iterazioni e power è la potenza del polinomio.
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power # man mano che iteer cresce, lr si riduce
    optimizer.param_groups[0]['lr'] = lr #cambia il lr dentro l'ottimizzatore
    return lr
    # return lr


# Costruisco una matrice di confusione per calcolare l'IoU (Intersection over Union) per ogni classe
def fast_hist(a, b, n):
    '''
    a (true label) and b (predicted) are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n) #crea una maschera per i valori di a che sono compresi tra 0 e n, (-1) usato come ignore label
    # a[k] è l'array di a che soddisfa la maschera 
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n) #conta quante volte si verifica ogni combinazione di classe vera e classe predetta, restituendo una matrice di confusione
# Calcolo l'IoU per ogni classe


#Calcola l'IoU (Intersection over Union) per ogni classe, usando la matrice di confusione calcolata sopra
def per_class_iou(hist):
    epsilon = 1e-5 #valore piccolo per evitare divisioni per zero
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
