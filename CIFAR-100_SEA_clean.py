import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time as time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch import Tensor
import random 
from torchvision import models
import statistics
import collections
import copy
import socket

transform = transforms.Compose(
    [transforms.ToTensor()
    ])

trainset = torchvision.datasets.CIFAR100(root='./', train=True,
                                        download=False, transform=transform)
Trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)


Trainloader_fake = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=False)

testset = torchvision.datasets.CIFAR100(root='./', train=False,
                                       download=False, transform=transform)

Testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

#========DEFINE THE CNN MODEL=====
starting_softmax_neurons = 20
feature_d = 1280
feature_w = 1
feature_h = 1
feature_extractor_size = feature_d*feature_w*feature_h

class CloudClassifier(nn.Module):
    def __init__(self):
        super(CloudClassifier, self).__init__()
        self.softmax_layer = nn.Linear(feature_extractor_size,starting_softmax_neurons) #2 output nodes initially
        self.drop1 = nn.Dropout(0.20)
    
    def forward(self, x):
        x = x.view(-1,feature_extractor_size) #FLATTENING OPERATION 1*1*512 IS OUTPUT AFTER THE PREVIOUS LAYER
        x = self.drop1(x)
        x = self.softmax_layer(x) #LAST LAYER DOES NOT NEED SOFTMAX BECAUSE THE LOSS FUNCTION WILL TAKE CARE OF IT
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cloud_cl = CloudClassifier()
cloud_cl = cloud_cl.to(device)

#FUNCTION TO CONVERT STRING BACK TO TENSOR (LABELS AND THE FEATURE MAPS)
def DecomposeString(Feature,batch_size):
    string_tensor = Feature
    listt = string_tensor.split(",")
    arr = np.array(listt,dtype = float)

    depth = int(arr[0])
    width = int(arr[1])
    height = int(arr[2])

    batch_size = int(arr.shape[0]/(depth*width*height))
    
    labels = []
    for i in range(3,3+batch_size):
        labels.append(arr[i])

    torch_labels = torch.from_numpy(np.array(labels,dtype=int))
    #remove first 3 elements from list
    arr = arr[3+batch_size:]

    torch_tensor = torch.from_numpy(arr)
    torch_tensor = torch_tensor.view(-1,depth,width,height)
    return torch_tensor,torch_labels

#Function to create artificial samples
def SMOTE(img_train, lab_train, total_new_points):
    length_of_dataset = len(img_train)
    number_of_points_to_be_chosen = 2 #distance can be calculated using only 2 points

    number_of_data_created = 0
    new_data_img = 0
    new_data_lab = []
    while(number_of_data_created != total_new_points):
        sequence = [i for i in range(length_of_dataset)]
        point_indices = random.sample(sequence, number_of_points_to_be_chosen) #choose any 2 training images from img_train
        
        p1 = img_train[point_indices[0]]
        p2 = img_train[point_indices[1]]

        new_lab = lab_train[point_indices[0]] #choose any point as it will have same index

        number_of_points = 5 if (total_new_points - number_of_data_created) > 5 else (total_new_points - number_of_data_created)
        for i in range(0,number_of_points):
            lamda = np.random.uniform(0,1)
            new_point = p1 + (lamda*(p2-p1))
            new_point = np.reshape(new_point,(1,feature_w,feature_h,feature_d))
            if(number_of_data_created == 0):
                new_data_img = new_point
                new_data_lab.append(new_lab)
            else:
                new_data_img = np.concatenate((new_data_img, new_point), axis=0)
                new_data_lab.append(new_lab)
            number_of_data_created += 1

    new_data_lab = np.asarray(new_data_lab)

    return new_data_img,new_data_lab

test_features_first_time = True
appending_message = False
training_features = 0
training_features_labels = 0
training_features_so_far = 0
training_features_labels_so_far = 0

testing_features = 0
testing_features_labels = 0
testing_features_so_far = 0
testing_features_labels_so_far = 0

ori_weight_message_string_length_list = []

#Variables for SMOTE
useful_images_list = []
useful_labels_list = []
length_of_dataset_after_filtration = []
acc_list = []

overall_training_time_list = []

training_time_list = []
weight_message_string_length = []
classification_time_list = []
append_time_list = []
smote_time_list = []

Exemplar_features = 0
Exemplar_labels = 0
first_time = 0
incremental_i = 0
cloud_total_nerons_in_final_layer = starting_softmax_neurons
max_samples_per_class_so_far = 0

def Cloud(Feature):
    global test_features_first_time
    global appending_message
    global training_features
    global training_features_labels
    global training_features_so_far
    global training_features_labels_so_far
    global testing_features
    global testing_features_labels
    global testing_features_so_far
    global testing_features_labels_so_far
    global ori_weight_message_string_length_list

    global training_time_list
    global weight_message_string_length
    global classification_time_list
    global append_time_list

    #Variables for SMOTE
    global useful_images_list
    global useful_labels_list
    global length_of_dataset_after_filtration

    global Exemplar_features
    global Exemplar_labels
    global first_time
    global incremental_i
    global acc_list
    global cloud_total_nerons_in_final_layer
    global cloud_cl

    global max_samples_per_class_so_far

    global overall_training_time_list
    global smote_time_list

    batch_size = 64

    cloud_criterion = nn.CrossEntropyLoss() #CROSS ENTROPY LOSS FUNCTION
    cloud_special_criterion = nn.CrossEntropyLoss(reduction='none')
    cloud_learning_rate = 0.0001
    cloud_training_epochs = 15
    cloud_optimizer = optim.Adam(cloud_cl.parameters(), lr=cloud_learning_rate) #ADAM OPTIMIZER

    Feature = Feature.replace(" ","") #remove any spaces
    task_character = Feature[-2] #Check the 2nd last character and decide what the program should do now
    print("task_character: "+str(task_character)+"\n")

    if(task_character == 't'): #train the model

        overall_training_time_tic = time.time()

        #Split 'k' and decompse string ony by one
        Feature = Feature[:-2]
        Feature_list = Feature.split('k') #every feature map is separated by 'k'

        Trainloader = 0
        #All features received from the IoT edge device
        for i in range(0,len(Feature_list)):
            inputs,labels = DecomposeString(Feature_list[i],batch_size)

            if(i==0):
                training_features = inputs
                training_features_labels = labels
            else:
                training_features = torch.cat([training_features,inputs],dim=0)
                training_features_labels = torch.cat([training_features_labels,labels],dim=0)
        
        training_features = training_features.float()
        training_features_labels = training_features_labels.float()

        #Joing all the tensors that were received due to RAM overload on IoT edge device
        if(appending_message == True): #This means IoT device has sent a message due to RAM overload
            training_features = torch.cat([training_features,training_features_so_far],dim=0)
            training_features_labels = torch.cat([training_features_labels,training_features_labels_so_far],dim=0)
            appending_message = False
        
        print("training_features before training: "+str(training_features.shape))
        training_features = np.reshape(training_features,(len(training_features),feature_w,feature_h,feature_d))

        #Exemplar loader
        exemplardataset = TensorDataset( Tensor(training_features), Tensor(training_features_labels) )
        Exemplarloader = DataLoader(exemplardataset, batch_size= batch_size, shuffle=False)

        if(first_time==0):

            length_of_dataset_after_filtration.append(len(training_features)) #LENGTH OF SAMPLES IN THE RESPECTIVE TRAINING ROUND
            useful_images_list.append(training_features)
            useful_labels_list.append(training_features_labels)
            
            traindataset_new = TensorDataset( Tensor(training_features), Tensor(training_features_labels))
            Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)
        else:
            #CARRY OUT SMOTE OVER HERE
            if(len(training_features)<=length_of_dataset_after_filtration[incremental_i-1]):
                print("\ncurrent incremental training round in minority ")
                if(len(training_features)<length_of_dataset_after_filtration[incremental_i-1]):
                    total_new_points = length_of_dataset_after_filtration[incremental_i-1] - len(training_features)
                    smote_tic = time.time()
                    remaining_images,remaining_labels = SMOTE(training_features, training_features_labels, total_new_points)
                    smote_toc = time.time()
                    smote_time = smote_toc - smote_tic
                    smote_time_list.append(smote_time)
                    #remaining_images = np.reshape(remaining_images,(total_new_points,feature_extractor_size,w,h))
                    remaining_images = np.reshape(remaining_images,(total_new_points,feature_w,feature_h,feature_d))

                    print("training_features shape: "+str(training_features.shape))
                    print("remaining_images shape: "+str(remaining_images.shape))
                    training_features = np.concatenate((training_features,remaining_images),axis=0)
                    training_features_labels = np.concatenate((training_features_labels,remaining_labels),axis=0)
                    traindataset = TensorDataset( Tensor(training_features), Tensor(training_features_labels) )
                    Trainloader_trail = DataLoader(traindataset, batch_size= batch_size, shuffle=False)
                    #get initial losses
                    traindataset_new = TensorDataset( Tensor(training_features), Tensor(training_features_labels))
                    Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)
                    #length_of_dataset_after_filtration[incremental_i-1] = len(training_features)

                    print("length of new images: "+str(len(training_features)))
                    length_of_dataset_after_filtration.append(len(training_features))
                    useful_images_list.append(training_features)
                    useful_labels_list.append(training_features_labels)
                else:
                    print("same number of images as last time")
                    length_of_dataset_after_filtration.append(len(training_features))
                    useful_images_list.append(training_features)
                    useful_labels_list.append(training_features_labels)

                    traindataset_new = TensorDataset( Tensor(training_features), Tensor(training_features_labels))
                    Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)
            elif(len(training_features)>length_of_dataset_after_filtration[incremental_i-1]):
                n_images_total = 0
                n_labels_total = 0
                print("length of useful_images_list: "+str(len(useful_images_list)))
                for kk in range(0,len(useful_images_list)):
                    total_new_points = len(training_features) - len(useful_images_list[kk])
                    smote_tic = time.time()
                    n_images,n_labels = SMOTE(useful_images_list[kk], useful_labels_list[kk], total_new_points)
                    smote_toc = time.time()
                    smote_time = smote_toc - smote_tic
                    smote_time_list.append(smote_time)
                    n_images = np.reshape(n_images,(total_new_points,feature_w,feature_h,feature_d))
                    print("useful_images_list[kk] shape: "+str(useful_images_list[kk].shape))
                    print("n_images(new artifical images) shape: "+str(n_images.shape))
                    useful_images_list[kk] = np.concatenate((useful_images_list[kk],n_images),axis=0)
                    useful_labels_list[kk] = np.concatenate((useful_labels_list[kk],n_labels),axis=0)
                    length_of_dataset_after_filtration[kk] = len(useful_images_list[kk])

                    if(kk==0):
                        n_images_total = n_images
                        n_labels_total = n_labels
                    else:
                        n_images_total = np.concatenate((n_images_total,n_images),axis=0)
                        n_labels_total = np.concatenate((n_labels_total,n_labels),axis=0)
                
                print("n_images_total(TOTAL artificial samples) shape: "+str(n_images_total.shape))
                print("current images_train shape: "+str(training_features.shape))
                n_images_total = np.concatenate((n_images_total,training_features),axis=0)
                n_labels_total = np.concatenate((n_labels_total,training_features_labels),axis=0)
                print("n_images_total(TOTAL appended samples) shape: "+str(n_images_total.shape))

                traindataset_new = TensorDataset( Tensor(n_images_total), Tensor(n_labels_total) )
                Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)
                

                length_of_dataset_after_filtration.append(len(training_features))
                useful_images_list.append(training_features)
                useful_labels_list.append(training_features_labels)

            current_features_train = 0
            current_features_label = 0

            #Get features of the current class
            for i, data in enumerate(Trainloader, 0):
                inputs, labels = data
                inputs = inputs.clone().detach()
                inputs = inputs.view(len(inputs),feature_d,feature_w,feature_h)
                labels = labels.long()

                Features = inputs
                #Features = Features.to(device)
                #labels = labels.to(device)

                if(i==0):
                    current_features_train = Features
                    current_features_label = labels
                else:
                    current_features_train = torch.cat([current_features_train,Features],dim=0)
                    current_features_label = torch.cat([current_features_label,labels],dim=0)

            #APPEND THE CURRENT SAMPLES FEATURES WITH THE EXEMPLARS
            appended_dataset_image = torch.cat([Exemplar_features,current_features_train],dim=0)
            appended_dataset_label = torch.cat([Exemplar_labels,current_features_label],dim=0)

            print("appended data shape is: "+str(appended_dataset_image.shape)+" and appended label shape is: "+str(appended_dataset_label.shape)+"\n")
            dataset = TensorDataset( appended_dataset_image, appended_dataset_label )
            Trainloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)

        print("data loader prepared \n")

        pre_training_cl = copy.deepcopy(cloud_cl)
        pre_training_weight_list = pre_training_cl.softmax_layer.weight.view(-1,feature_extractor_size*cloud_total_nerons_in_final_layer).cpu().detach().numpy().tolist()[0]
        pre_training_bias_list = pre_training_cl.softmax_layer.bias.view(-1,cloud_total_nerons_in_final_layer).cpu().detach().numpy().tolist()[0]

        print("before training, cloud_cl is: "+str(cloud_cl))
        training_time_tic = time.time()
        #Train all the features on the cloud now
        for epoch in range(0,cloud_training_epochs):
            for i, data in enumerate(Trainloader, 0):

                inputs, labels = data

                #Pass the data now to the neural network
                inputs = inputs.clone().detach()
                inputs = inputs.view(len(inputs),feature_d,feature_w,feature_h)
                labels = labels.long()

                inputs=inputs.to(device)
                labels=labels.to(device)

                outputs = cloud_cl.forward(inputs)
                loss = cloud_criterion(outputs, labels)
                        
                cloud_optimizer.zero_grad()
                loss.backward()
                cloud_optimizer.step()

        training_time_toc = time.time()
        overall_training_time_toc = time.time()
        training_time = (training_time_toc-training_time_tic)
        training_time_list.append(training_time)

        overall_training_time = overall_training_time_toc - overall_training_time_tic
        overall_training_time_list.append(overall_training_time)

        training_features = 0
        training_features_labels = 0
        first_time = 1
        

        for i, data in enumerate(Exemplarloader, 0):
            inputs, labels = data

            inputs = inputs.clone().detach()
            inputs = inputs.view(len(inputs),feature_d,feature_w,feature_h)
            labels = labels.long()

            Features = inputs
            if(i==0 and incremental_i==0):
                with torch.no_grad():
                    Exemplar_features = Features
                    Exemplar_labels = labels
            else:
                with torch.no_grad():
                    Exemplar_features = torch.cat([Exemplar_features,Features],dim=0)
                    Exemplar_labels = torch.cat([Exemplar_labels,labels],dim=0)

        incremental_i += 1 #INCREMENTAL TRAINING ROUND COUNTER

    elif(task_character == 'c'): #time for classification (overall accuracy)
        classification_time_tic = time.time()
        cloud_cl.eval() #put the model is evaluation mode

        #Split 'k' and decompse string ony by one
        Feature = Feature[:-2]
        Feature_list = Feature.split('k')


        for i in range(0,len(Feature_list)):
            inputs,labels = DecomposeString(Feature_list[i],batch_size)
            
            testing_features = inputs
            testing_features_labels = labels

            testing_features = testing_features.float()
            testing_features_labels = testing_features_labels.float()

            if(i == 0 and test_features_first_time == True): #The very first time we are processing an append message
                test_features_first_time = False
                testing_features_so_far = testing_features
                testing_features_labels_so_far = testing_features_labels
            else:
                testing_features_so_far = torch.cat([testing_features_so_far,testing_features],dim=0)
                testing_features_labels_so_far = torch.cat([testing_features_labels_so_far,testing_features_labels],dim=0)

        print("testing_features_so_far shape is: "+str(testing_features_so_far.shape))
        testdatasett = TensorDataset( testing_features_so_far, testing_features_labels_so_far )
        Testloaderr = DataLoader(testdatasett, batch_size= batch_size, shuffle=False)

        correct = 0
        total = 0
        top = 5
        print(cloud_cl)
        targets = []
        predictions = []
        with torch.no_grad():
            for data in Testloaderr:
                images, labels = data

                images = images.clone().detach()
                images = images.view(len(images),feature_d,feature_w,feature_h) #RESHAPE THE IMAGES
                labels = labels.long() #MUST CONVERT LABEL TO LONG FORMAT

                images=images.to(device)
                labels=labels.to(device)

                #outputs = squeezenet.layers(images)
                outputs = cloud_cl.forward(images)

                #COMPUTE TOP-5 accuracy
                _,pred=outputs.topk(top,1,largest=True,sorted=True) #select top 5
                total += labels.size(0)

                _,predd=outputs.topk(1,1,largest=True,sorted=True) #select top 5
                for j in range(len(pred)):
                    targets.append(labels[j].item())
                    predictions.append(predd[j].item())

                    tensor = pred[j]
                    true_label = labels[j]
                    for k in range(len(tensor)):
                        if(true_label == tensor[k].item()):
                            correct += 1
                            #predictions.append()
                            break
        if(total>0):
            acc = float(correct/total)*100.0
            print("correct is: "+str(correct)+"\n")
            print("total is: "+str(total)+"\n")
            print("accuracy is: "+str(acc)+"\n")
            acc_list.append(acc)

        #from sklearn.metrics import confusion_matrix
        #from resources.plotcm import plot_confusion_matrix
        #cm = confusion_matrix(targets, predictions)

        print("targets: "+str(targets))
        print("predictions: "+str(predictions))

        cloud_cl.train()

        #ADD NEURONS TO THE FINAL SOFTMAX CLASSIFICATION LAYER
        print("modifying the cloud model now \n")
        
        #Add nerons to the fina layer and set the new random weights
        current_weights = cloud_cl.softmax_layer.weight.data
        current_bias = cloud_cl.softmax_layer.bias.data

        cloud_total_nerons_in_final_layer = cloud_total_nerons_in_final_layer + starting_softmax_neurons
        cloud_cl.softmax_layer = nn.Linear(feature_extractor_size,cloud_total_nerons_in_final_layer)

        print("cloud_total_nerons_in_final_layer: "+str(cloud_total_nerons_in_final_layer))

        new_weights = -0.0001 * np.random.random_sample((starting_softmax_neurons, feature_extractor_size)) + 0.0001 #RANDOMLY INITIALIZED WEIGHTS FOR THE NEW TASKS
        new_weights = torch.from_numpy(new_weights).float() #CONVERT NEWLY RANDONLY INITIALIZED WEIGHTS TO A TORCH TENSOR
        new_weights = new_weights.to(device)
        new_weights = torch.cat([current_weights,new_weights],dim=0) #Concatenate the new initialized weights with the old weights for old task
        cloud_cl.softmax_layer.weight.data.copy_(torch.tensor(new_weights, requires_grad=True))

        new_biases = np.zeros(starting_softmax_neurons)
        new_biases = torch.from_numpy(new_biases).float() #CONVERT NEWLY RANDONLY INITIALIZED WEIGHTS TO A TORCH TENSOR
        new_biases = new_biases.to(device)
        new_biases = torch.cat([current_bias,new_biases],dim=0)
        cloud_cl.softmax_layer.bias.data.copy_(torch.tensor(new_biases, requires_grad=True))

        #After adding neurons to the final layer, the network needs to be pushed to the GPU again
        cloud_cl = cloud_cl.to(device)
        cloud_optimizer = optim.Adam(cloud_cl.parameters(), lr=cloud_learning_rate) #UPDATE OPTIMIZER

        classification_time_toc = time.time()
        classification_time = (classification_time_toc - classification_time_tic)
        classification_time_list.append(classification_time)
    
    elif(task_character == 'a'): #just append the tensors and send back acknowledgement back to IoT edge device

        append_tic = time.time()
        #Split 'k' and decompse string ony by one
        Feature = Feature[:-2]
        Feature_list = Feature.split('k') #every feature map is separated by 'k'

        print("start appending \n")
        for i in range(0,len(Feature_list)):
            inputs,labels = DecomposeString(Feature_list[i],batch_size)
            
            training_features = inputs
            training_features_labels = labels

            training_features = training_features.float()
            training_features_labels = training_features_labels.float()

            if(i == 0 and appending_message == False): #The very first time we are processing an append message
                training_features_so_far = training_features
                training_features_labels_so_far = training_features_labels
            else:
                training_features_so_far = torch.cat([training_features_so_far,training_features],dim=0)
                training_features_labels_so_far = torch.cat([training_features_labels_so_far,training_features_labels],dim=0)

        print("training_features_so_far shape: "+str(training_features_so_far.shape))
        appending_message = True
        append_toc = time.time()
        append_time_list.append(append_toc-append_tic)

        if(len(training_features_so_far) > max_samples_per_class_so_far):
            number_of_digits = len(str(len(training_features_so_far)))
            test_number = str(len(training_features_so_far))[0]
            for c in range(number_of_digits-1):
                test_number += "0"
            test_number = int(test_number)
            max_samples_per_class_so_far = len(training_features_so_far) #This is the new maximum now

    elif(task_character == 'd'): #means process done
        print("============================CLOUD RESULTS=====================================")
        print("number of classes being learnt at a time: "+str(starting_softmax_neurons))
        print("cloud training_time_list: "+str(training_time_list))
        print("TOTAL TRAINING TIME (SECONDS): "+str(sum(training_time_list)))
        print("append_time_list: "+str(append_time_list))
        print("TOTAL append_time_list: "+str(sum(append_time_list)))
        print("acc_list: "+str(acc_list))
        print("overall_training_time_list: "+str(overall_training_time_list))
        print("TOTAL overall_training_time_list: "+str(sum(overall_training_time_list)))
        print("smote_time_list: "+str(smote_time_list))
        print("TOTAL smote_time_list: "+str(sum(smote_time_list)))
#======================================CLOUD CODE END HERE========================================






#======================================EDGE CODE DOWN HERE========================================
class EdgeClassifier(nn.Module):
    def __init__(self):
        super(EdgeClassifier, self).__init__()
        self.softmax_layer = nn.Linear(feature_extractor_size,starting_softmax_neurons) #2 output nodes initially
        self.drop1 = nn.Dropout(0.25)
    
    def forward(self, x):
        x = x.view(-1,feature_extractor_size) #FLATTENING OPERATION 1*1*512 IS OUTPUT AFTER THE PREVIOUS LAYER
        x = self.drop1(x)
        x = self.softmax_layer(x) #LAST LAYER DOES NOT NEED SOFTMAX BECAUSE THE LOSS FUNCTION WILL TAKE CARE OF IT
        return x

class EdgeTempClassifier(nn.Module):
    def __init__(self):
        super(EdgeTempClassifier, self).__init__()
        self.drop2 = nn.Dropout(0.25)
        self.softmax_layer = nn.Linear(feature_extractor_size,starting_softmax_neurons)
    
    def forward(self, x):
        x = x.view(-1,feature_extractor_size) #FLATTENING OPERATION 1*1*512 IS OUTPUT AFTER THE PREVIOUS LAYER
        x = self.drop2(x)
        x = self.softmax_layer(x) #LAST LAYER DOES NOT NEED SOFTMAX BECAUSE THE LOSS FUNCTION WILL TAKE CARE OF IT
        return x

def TorchNormalizedTrainingData():
    img_list = []
    lab_list = []
    with torch.no_grad():
        for data in Trainloader_fake:
            images, labels = data
            for j in range(0,len(images)):
              img_list.append(np.array(images[j]))
              lab_list.append(int(labels[j]))

    img_list = np.array(img_list)    
    lab_list = np.array(lab_list)
    #Sort training dataset wrt to labels

    sorted_X_train = [] #this will contains training images sorted based on the label number
    sorted_Y_train = [] #label
    start_index = 0
    end_index = 100
    for j in range(start_index,end_index):
      searchval = j
      all_positions_of_label = np.where(lab_list == searchval)[0] #Find all indexes that contains `searchval`
      for j in range(0,len(all_positions_of_label)):
          image = img_list[all_positions_of_label[j]]
          sorted_X_train.append(image)
          sorted_Y_train.append(searchval)
    
    sorted_X_train = np.array(sorted_X_train)
    sorted_X_train = sorted_X_train.reshape(50000, 32, 32, 3)

    return sorted_X_train,sorted_Y_train

def TorchNormalizedTestingData():
    img_list = []
    lab_list = []
    with torch.no_grad():
      for data in Testloader:
        images, labels = data
        for j in range(0,len(images)):
          img_list.append(np.array(images[j]))
          lab_list.append(int(labels[j]))

    img_list = np.array(img_list)    
    lab_list = np.array(lab_list)
    #Sort training dataset wrt to labels

    sorted_X_test = [] #this will contains training images sorted based on the label number
    sorted_Y_test = [] #label
    start_index = 0
    end_index = 100
    for j in range(start_index,end_index):
      searchval = j
      all_positions_of_label = np.where(lab_list == searchval)[0] #Find all indexes that contains `searchval`
      for j in range(0,len(all_positions_of_label)):
        image = img_list[all_positions_of_label[j]]
        sorted_X_test.append(image)
        sorted_Y_test.append(searchval)
    sorted_X_test = np.array(sorted_X_test)
    sorted_X_test = sorted_X_test.reshape(10000, 32, 32, 3)
    
    return sorted_X_test,sorted_Y_test

#ANY DATA RECEIVED BY THE CLOUD WILL BE TAKEN CARE OF BY THIS FUNCTION
def ReceiveDataFromCloud(client):
    recevied_data = " "
    received_result = ""
    print("receving start")
    while recevied_data[-1]!="!":
        data = client.recv(10000).decode("utf-8")
        recevied_data += data
        received_result += data
    client.close()
    return received_result

#FUNCTION FOR CONNECTED TO GOOGLE CLOUD VM AND TRANSMITTING DATA
def SendDataToCloud(Feature):

    #Connect to the cloud and send the data
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)#AF_NET = ipv4, SOCK_STREAM = tcp oriented
    
    #ip and port of the virtual machine server need to be configured
    ip = '' 
    port = 0
    address = (ip,port) #address of the server
    client.connect(address)
    print("sending to the cloud \n")
    
    #Send the string to the cloud
    client.send(Feature.encode())
    print("sent to the cloud \n")

    #Receive response from the server and only then continue
    received_string = ReceiveDataFromCloud(client)
    print("recvd from the cloud \n")
    return received_string

#FUNCTION FOR SENDING TEST DATASET TO THE CLOUD FOR COMPUTING CLASSIFICATION ACCURACY ON TEST DATASET
def CalculateModelAccuracy(squeezenet,device,Testloader,batch_size):
    
    overall_string_tensor = ""
    with torch.no_grad():
        for data in Testloader:
            images, labels = data

            images = images.clone().detach()
            images = images.view(len(images),3,32,32) #RESHAPE THE IMAGES
            labels = labels.long() #MUST CONVERT LABEL TO LONG FORMAT

            images=images.to(device)
            labels=labels.to(device)

            outputs = squeezenet.layers(images)

            #Convert the tensor to string
            string_tensor = TensorToString(labels,outputs)
            overall_string_tensor = overall_string_tensor + string_tensor + "k"

        #Send the string to the cloud
        overall_string_tensor = overall_string_tensor[:-1] + "c!"
        Cloud(overall_string_tensor)
    
def GetTempClassifierAccuracy(outputs, labels):
    acc = torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item() / len(labels)
    return acc

def GetLosses(Trainloader_trail,device,squeezenet,cl):
    initial_loss_list = []
    edge_special_criterion = nn.CrossEntropyLoss(reduction='none')
    image_loss_dictionary = collections.defaultdict(list)
    label_loss_dictionary = collections.defaultdict(list)

    for _, data in enumerate(Trainloader_trail, 0):
        inputs, labels = data

        inputs = inputs.clone().detach()
        inputs = inputs.view(len(inputs),3,32,32)
        labels = labels.long()
                
        inputs=inputs.to(device)
        labels=labels.to(device)

        with torch.no_grad():
            Feature_extractor_outputs = squeezenet.layers(inputs)
            print("Feature_extractor_outputs shape: "+str(Feature_extractor_outputs.shape))
            outputs = cl.forward(Feature_extractor_outputs)
            loss = edge_special_criterion(outputs, labels)
            for j in range(0,len(loss)):
                initial_loss_list.append(loss[j].item())
                image_loss_dictionary[loss[j].item()].append(Feature_extractor_outputs[j])
                label_loss_dictionary[loss[j].item()].append(labels[j].cpu().numpy())

    print("initial_loss_list: "+str(initial_loss_list))
    return initial_loss_list, image_loss_dictionary, label_loss_dictionary#, acc_loss_dictionary

#FUNCTION FOR COMPUTING THE STATISTICS OF THE SORTED LOSS VALUES OF THE TRAINING IMAGES
def GetLossStatistics(sorted_initial_list):
    #Calculate mean
    mean = statistics.mean(sorted_initial_list)

    #Calculate median
    median = statistics.median(sorted_initial_list)

    #Calculate standard deviation
    standard_deviation = statistics.stdev(sorted_initial_list)
    
    return mean, median, standard_deviation

#FUNCTION FOR FINDING THE CUTOFF LOSS VALUE AND DISCARDING THE SAMPLES
def Cutoff(mean,median,standard_deviation,image_loss_dictionary,label_loss_dictionary,depth,width,height,divider):
    sorted_loss_list = sorted(image_loss_dictionary)
    number_of_samples_to_be_dropped = 0
    
    cutoff_loss_value = min(sorted_loss_list)
    min_loss_value = min(sorted_loss_list)
    for i in range(0,len(sorted_loss_list)):
        distance = median - sorted_loss_list[i]
        if(distance < standard_deviation):
            cutoff_loss_value = sorted_loss_list[i]
            print("min loss: "+str(min_loss_value)+", distance: "+str(distance)+" , std: "+str(standard_deviation))
            break
    
    #cutoff_loss_value = cutoff_loss_value - (0.25*(cutoff_loss_value-min_loss_value))
    number_of_samples_to_be_dropped = 0
    for i in range(0,len(sorted_loss_list)):
        current_loss = sorted_loss_list[i]
        if(current_loss < cutoff_loss_value):
            number_of_samples_to_be_dropped += 1
        else:
            break

    omega = 1.00
    number_of_samples_to_be_dropped  = int(omega * number_of_samples_to_be_dropped)
    print("number_of_samples_to_be_dropped: "+str(number_of_samples_to_be_dropped))

    number_of_samples_to_be_selected = 0
    #randomly select losses
    if(number_of_samples_to_be_dropped > 0):
        number_of_samples_to_be_selected = len(sorted_loss_list) - number_of_samples_to_be_dropped
        print("number_of_samples_to_be_selected: "+str(number_of_samples_to_be_selected))
        randomly_selected_losses = random.sample(sorted_loss_list, k=number_of_samples_to_be_selected)
    else:
        randomly_selected_losses = random.sample(sorted_loss_list, k=len(sorted_loss_list))
        print("number_of_samples_to_be_selected: "+str(len(sorted_loss_list)))
        print("must select all the samples now!!!!! ")

    #get all the images and labels above the cutoff value
    number_of_samples_to_selected = 0
    new_images = 0
    new_labels = []
    filtered_original_losses = []
    
    
    #RANDOM SAMPLING
    #"""
    for i in range(0,len(randomly_selected_losses)):
        key_loss = randomly_selected_losses[i]
        selected_image = image_loss_dictionary[key_loss]
        selected_label = label_loss_dictionary[key_loss]

        if(number_of_samples_to_selected == 0):
            for j in range(0,len(selected_image)): # aloss key can have duplicates
                if(j==0):
                    new_images = selected_image[j]
                else:
                    new_images = torch.cat([new_images,selected_image[j]],dim=0)
                new_labels.append(selected_label[j])
                filtered_original_losses.append(key_loss)
        else:
            for j in range(0,len(selected_image)): # aloss key can have duplicates
                new_images = torch.cat([new_images,selected_image[j]],dim=0)
                new_labels.append(selected_label[j])
                filtered_original_losses.append(key_loss)
        number_of_samples_to_selected += len(selected_image)
    
    new_images = new_images.reshape((number_of_samples_to_selected,depth,width,height))
    print("new_images shape: "+str(new_images.shape))

    new_labels = torch.from_numpy(np.asarray(new_labels)).float()
    print("new_labels shape: "+str(new_labels.shape))

    return new_images,new_labels

#MAIN FUNCTION FOR PERFORMING DATA SAMPLING
def GetusefulData(images_train,labels_train,batch_size,squeezenet,cl,divider):

    traindataset = TensorDataset( Tensor(images_train), Tensor(labels_train) )
    Trainloader_trail = DataLoader(traindataset, batch_size= batch_size, shuffle=False)

    #Do single forward pass and get the losses
    initial_loss_list, image_loss_dictionary, label_loss_dictionary = GetLosses(Trainloader_trail,device,squeezenet,cl)
    sorted_initial_list = sorted(initial_loss_list)
    mean, median, standard_deviation = GetLossStatistics(sorted_initial_list)
    new_images, new_labels = Cutoff(mean,median,standard_deviation,image_loss_dictionary,label_loss_dictionary,feature_d,feature_w,feature_h,divider)
    

    rejection_rate = (1.0 - (len(new_images)/len(images_train)))*100.0
    print("rejection rate: "+str(rejection_rate))
    return new_images.cpu().numpy(), new_labels.cpu().numpy()



#FUNCTION TO CONVERT TENSOR TO STRING
def TensorToString(labels,outputs):

    tensor = outputs
    tensor = tensor.view(-1,outputs.shape[1]*outputs.shape[2]*outputs.shape[3]) #flatten the tensor

    string_tensor = ""
    string_tensor = str(outputs.shape[1]) + "," + str(outputs.shape[2]) + "," + str(outputs.shape[3]) + ","
    mini_batch_length = len(labels)
    for i in range(0,mini_batch_length):
        string_tensor = string_tensor + str(labels[i].item()) + ","


    new_string = string_tensor
    for i in range(0,mini_batch_length):
        numpy_tensor = tensor[i].cpu().numpy()
        out_string = ','.join(str(x) for x in numpy_tensor)
        new_string = new_string + out_string + ","
    new_string = new_string[:-1] #remove last comma
    #new_string = new_string + "t!" 

    string_tensor = new_string
    return string_tensor

#FUNCTION TO DECODE WEIGHTS AND BIASES (SEPARATING WEIGHTS & BIASES FROM COMMAS)
def DecodeWeightsAndBiases(message):
    task_character = message[-2]

    bias_string = message[message.find("b")+2:message.find(task_character)-1]
    bias_list = list(map(float, bias_string.split(",")))

    weight_string = message[0:message.find("i")-1]
    weight_list = list(map(float, weight_string.split(","))) 

    indices_string = message[message.find("i")+2:message.find("b")-1]
    indices_list = indices_string.split(",")

    return weight_list, bias_list, indices_list, task_character


#==========MAIN PROGRAM==========

Images_train, Labels_train = TorchNormalizedTrainingData()
Images_test, Labels_test = TorchNormalizedTestingData()

squeezenet = models.mnasnet1_0(pretrained=True)
squeezenet = squeezenet.to(device)
squeezenet.eval()

edge_cl = EdgeClassifier()
edge_cl = edge_cl.to(device) #MAP THE MODEL ONTO THE GPU
edge_cl.eval()

edge_temp_cl = EdgeTempClassifier()
edge_temp_cl = edge_temp_cl.to(device) #MAP THE MODEL ONTO THE GPU
edge_temp_cl.eval()

edge_criterion = nn.CrossEntropyLoss() #CROSS ENTROPY LOSS FUNCTION
edge_optimizer = optim.Adam(edge_cl.parameters(), lr=1e-4) #ADAM OPTIMIZER


#START TRAINING
number_of_classes = 100
batch_size = 128
class_start_index = 0
number_of_classes_to_be_added_ever_increment = starting_softmax_neurons
total_nerons_in_final_layer = number_of_classes_to_be_added_ever_increment
number_of_training_images_per_class = 500 #THIS IS FIXED FOR CIFAR-100 DATASET
number_of_test_images_per_class = 100 #THIS IS FIXED FOR CIFAR-100 DATASET
number_of_increments = int(number_of_classes/number_of_classes_to_be_added_ever_increment)
time_list = []
test_time_list = [] # TIME TAKEN FOR SENDING TESTLOADER TO THE CLOUD
datasize_sent_list = [] #FOR KEEPING TRACK OF THE SIZE OF THE DATA THAT HAS BEEN SENT TO THE CLOUD
edge_processing_time_list = []
samples_transmitted = []
incremental_time_list = []
samples_transmitted_count = 0
total_append_operation_time = 0

mini_batch_counter = 0
t0 = 0 # time taken to process first mini batch
length_so_far = 0 #Length of the string so far
number_of_overloads = 0
divider = 0.5

depth = feature_d
w = 1
h = 1

number_of_increments = number_of_classes
#number_of_increments = 2*starting_softmax_neurons #REMOVE LATER!!

total_classes_processed = 0
total_train_samples_processed = 500
total_test_samples_processed = 100
t_counter = 0 #training round counter

overall_tic=time.time()
for edge_incremental_i in range(0,number_of_increments):#EVERY INCREMENTAL STEP WE WILL LEARN 2 CLASSES AT A TIME

    incremental_tic = time.time()
    #GET TRAINING DATA (NEW TRAINING DATA EVERY INCREMENT)
    images_train = Images_train[0:number_of_training_images_per_class]
    labels_train = Labels_train[0:number_of_training_images_per_class]

    #Carry out data sampling here
    edge_tic = time.time()
    fc_layers = edge_cl
    if(t_counter > 0):
        fc_layers = edge_temp_cl
        for kk in range(0,len(labels_train)):
            labels_train[kk] -= (number_of_classes_to_be_added_ever_increment*t_counter)
    fc_layers.eval()
    new_images, new_labels = GetusefulData(images_train, labels_train, batch_size, squeezenet, fc_layers,divider)
    fc_layers.train()
    if(t_counter > 0):
        for kk in range(0,len(new_labels)):
            new_labels[kk] += (number_of_classes_to_be_added_ever_increment*t_counter)
    traindataset_new = TensorDataset( Tensor(new_images), Tensor(new_labels))
    Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)

    Trainloader_iterations = 0
    if(len(new_images) % batch_size == 0):
        Trainloader_iterations = len(new_images) // batch_size
    else:
        Trainloader_iterations = (len(new_images) // batch_size) + 1
    edge_toc = time.time()
    edge_processing_time_list.append(edge_toc-edge_tic)

    samples_transmitted_count += len(new_images)
    print("len(new_images): "+str(len(new_images)))

    overall_string_tensor = ""
    for i, data in enumerate(Trainloader, 0): #This will only run once i.e. only 1 epoch
        
        mini_batch_tic = time.time()
        inputs, labels = data

        inputs = inputs.clone().detach()
        inputs = inputs.view(len(inputs),depth,w,h)
        labels = labels.long()

        inputs=inputs.to(device)
        labels=labels.to(device)

        outputs = inputs

        #Send the data to the cloud device (images and the labels)
        string_tensor = TensorToString(labels,outputs)
        overall_string_tensor = overall_string_tensor + string_tensor + "k"
        mini_batch_toc = time.time()
        time_diff = mini_batch_toc - mini_batch_tic

        if(i==0):
            t0 = time_diff
        else:
            if((time_diff - t0) > 0.5): #RAM overload, transmit to the cloud asap and clear string on IoT edge device
                print("RAM overload!")
                if(i == Trainloader_iterations - 1): #RAM overload at the last iteration! so start training already
                    overall_string_tensor = overall_string_tensor[:-1] + "t!"
                    append_ticc = time.time()
                    Cloud(overall_string_tensor)
                    append_tocc = time.time()
                    total_append_operation_time += (append_tocc-append_ticc)
                else:
                    overall_string_tensor = overall_string_tensor[:-1] + "a!" #remove the last k and add a! at the end
                    append_ticc = time.time()
                    Cloud(overall_string_tensor)
                    append_tocc = time.time()
                    total_append_operation_time += (append_tocc-append_ticc)
                length_so_far += len(overall_string_tensor)
                length_so_far *= 1e-6
                datasize_sent_list.append(length_so_far)
                overall_string_tensor = ""
                number_of_overloads += 1

    #Clear off the memory by deleting the first few samples in the list
    train_index = [v for v in range(total_train_samples_processed)]
    Images_train = np.delete(Images_train, train_index, axis = 0)
    del Labels_train[:total_train_samples_processed]

    #Send to cloud and modify the edge classifier
    if(total_classes_processed == number_of_classes_to_be_added_ever_increment - 1): #Send to cloud with the training message
        t_counter += 1
        total_classes_processed = -1
        if(len(overall_string_tensor)>1):
            overall_string_tensor = overall_string_tensor[:-1] + "t!" #remove the last k and add t! at the end
            length_so_far += len(overall_string_tensor)
            length_so_far *= 1e-6
            datasize_sent_list.append(length_so_far)
            Cloud(overall_string_tensor)        
        overall_string_tensor = ""

        samples_transmitted.append(samples_transmitted_count)
        samples_transmitted_count = 0

        #SEND TESTLOADER TO THE CLOUD FOR COMPUTING ACCURACY
        CalculateModelAccuracy_time_tic = time.time()

        images_test = Images_test[0:(number_of_test_images_per_class*number_of_classes_to_be_added_ever_increment)]
        labels_test = Labels_test[0:(number_of_test_images_per_class*number_of_classes_to_be_added_ever_increment)]
        
        testdataset = TensorDataset( Tensor(images_test), Tensor(labels_test) )
        Testloader = DataLoader(testdataset, batch_size= batch_size, shuffle=False)
        
        test_index = [v for v in range(number_of_test_images_per_class*number_of_classes_to_be_added_ever_increment)]
        Images_test = np.delete(Images_test, test_index, axis = 0)
        del Labels_test[:(number_of_test_images_per_class*number_of_classes_to_be_added_ever_increment)]

        CalculateModelAccuracy(squeezenet,device,Testloader,batch_size)
        CalculateModelAccuracy_time_toc = time.time()
        testloader_time = CalculateModelAccuracy_time_toc - CalculateModelAccuracy_time_tic
        test_time_list.append(testloader_time)

        print("modify softmax layer edge "+str(edge_cl))
        #Add neurons to the SoftMax classification layer
        
        current_weights = edge_cl.softmax_layer.weight.data
        current_bias = edge_cl.softmax_layer.bias.data

        new_total_nerons_in_final_layer = total_nerons_in_final_layer + number_of_classes_to_be_added_ever_increment
        total_nerons_in_final_layer += number_of_classes_to_be_added_ever_increment
        edge_cl.softmax_layer = nn.Linear(feature_extractor_size,total_nerons_in_final_layer)

        new_weights = -0.0001 * np.random.random_sample((number_of_classes_to_be_added_ever_increment, feature_extractor_size)) + 0.0001 #RANDOMLY INITIALIZED WEIGHTS FOR THE NEW TASKS
        new_weights = torch.from_numpy(new_weights).float() #CONVERT NEWLY RANDONLY INITIALIZED WEIGHTS TO A TORCH TENSOR
        new_weights = new_weights.to(device)
        new_weights = torch.cat([current_weights,new_weights],dim=0) #Concatenate the new initialized weights with the old weights for old task
        edge_cl.softmax_layer.weight.data.copy_(torch.tensor(new_weights, requires_grad=True))

        new_biases = np.zeros(number_of_classes_to_be_added_ever_increment)
        new_biases = torch.from_numpy(new_biases).float() #CONVERT NEWLY RANDONLY INITIALIZED WEIGHTS TO A TORCH TENSOR
        new_biases = new_biases.to(device)
        new_biases = torch.cat([current_bias,new_biases],dim=0)
        edge_cl.softmax_layer.bias.data.copy_(torch.tensor(new_biases, requires_grad=True))

        #After adding neurons to the final layer, the network needs to be pushed to the GPU again
        edge_cl = edge_cl.to(device)


        #Assign random weights to tempcl
        edge_modifiable_cl = copy.deepcopy(edge_cl)
        edge_modifiable_cl_weight = edge_modifiable_cl.softmax_layer.weight[(total_nerons_in_final_layer - number_of_classes_to_be_added_ever_increment):total_nerons_in_final_layer] #base weights
        edge_modifiable_cl_bias = edge_modifiable_cl.softmax_layer.bias[(total_nerons_in_final_layer - number_of_classes_to_be_added_ever_increment):total_nerons_in_final_layer] #base biases
        edge_temp_cl.softmax_layer.weight.data.copy_(torch.tensor(edge_modifiable_cl_weight, requires_grad=True))
        edge_temp_cl.softmax_layer.bias.data.copy_(torch.tensor(edge_modifiable_cl_bias, requires_grad=True))

    else: #Send to cloud with the append message
        if(len(overall_string_tensor)>1):
            overall_string_tensor = overall_string_tensor[:-1] + "a!" #remove the last k and add a! at the end
            length_so_far += len(overall_string_tensor)
            length_so_far *= 1e-6
            datasize_sent_list.append(length_so_far)
            append_ticc = time.time()
            Cloud(overall_string_tensor)
            append_tocc = time.time()
            total_append_operation_time += (append_tocc-append_ticc)
        overall_string_tensor = ""

    total_classes_processed += 1
    incremental_toc = time.time()
    incre_time = incremental_toc - incremental_tic
    incremental_time_list.append(incre_time)

  
overall_toc=time.time()
stop_cloud_message = "d!"
Cloud(stop_cloud_message)
overall_time = (overall_toc-overall_tic)-sum(test_time_list)

print("======================EDGE RESULTS=========================")
print("Number 0f classes learnt at a time: "+str(starting_softmax_neurons))
print("incremental_time_list: "+str(incremental_time_list))
print("overall time: "+str(sum(incremental_time_list)))
print("total_append_operation_time: "+str(total_append_operation_time))
print("number_of_overloads: "+str(number_of_overloads))
print("edge processing time (data sampling and forward pass): "+str(edge_processing_time_list))
print("TOTAL edge processing time: ")
print(sum(edge_processing_time_list))
print("testing time: ")
print(test_time_list)
print("TOTAL testing time: ")
print(sum(test_time_list))

print("amount of data sent so far (MB): ")
print(datasize_sent_list)
print("TOTAL data sent (MB): ")
print(sum(datasize_sent_list))

print("samples_transmitted: "+str(samples_transmitted))
print("TOTAL samples_transmitted: "+str(sum(samples_transmitted)))

base_samples = number_of_classes_to_be_added_ever_increment * number_of_training_images_per_class
rejection_rate = []
for i in range(len(samples_transmitted)):
    difference = base_samples - samples_transmitted[i]
    original_amount = base_samples
    decrease_percentage = (difference/original_amount)*100.0
    rejection_rate.append(decrease_percentage)
print("rejection_rate: "+str(rejection_rate))
