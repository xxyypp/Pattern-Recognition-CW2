from sklearn.utils import shuffle

train_unique_labels = np.unique(lbl_train)

# pick 100 of people and move all pictures of those people to your validation set
shuffle_lbl = shuffle(train_unique_labels)[:100] 

val = np.zeros((1,2049))
train = np.vstack((fea_train.T, lbl_train)).T

for identity in tqdm( shuffle_lbl ):
    #Select validation set from training data
    validation = train[ np.where(train[:,-1]==identity)]
    
    val = np.vstack( ( val, validation ) )
    
    train = train[ np.where( train[ :, -1 ] != identity )]
val = np.delete(val, (0), axis=1)
