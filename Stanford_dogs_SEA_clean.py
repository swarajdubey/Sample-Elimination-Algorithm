import torch
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
import os
from PIL import Image
from scipy.io import loadmat

transform = transforms.Compose(
    [transforms.ToTensor()
    ])

#========DEFINE THE CNN MODEL=====
starting_softmax_neurons = 40
feature_d = 1280
feature_w = 3
feature_h = 3

feature_extractor_size = feature_d*feature_w*feature_h
class CloudClassifier(nn.Module):
    def __init__(self):
        super(CloudClassifier, self).__init__()
        self.softmax_layer = nn.Linear(feature_extractor_size,starting_softmax_neurons) #2 output nodes initially
        self.drop1 = nn.Dropout(0.30)
    
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


def GetTrainingAccuracy(cl,device,Trainloader):
    cl.eval()
    correct = 0
    total = 0
    top = 5
    with torch.no_grad():
        for data in Trainloader:
            images, labels = data

            images = images.clone().detach()
            images = images.view(len(images),feature_d,feature_w,feature_h) #RESHAPE THE IMAGES
            labels = labels.long() #MUST CONVERT LABEL TO LONG FORMAT

            images=images.to(device)
            labels=labels.to(device)

            outputs = cl.forward(images)

            #COMPUTE TOP-5 accuracy
            _,pred=outputs.topk(top,1,largest=True,sorted=True) #select top 5
            total += labels.size(0)

            for j in range(len(pred)):
                tensor = pred[j]
                true_label = labels[j]
                for k in range(len(tensor)):
                    if(true_label == tensor[k].item()):
                        correct += 1
                        break

    acc = float(correct/total)*100.0
    cl.train()
    return acc


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

#Variables for SMOTE
useful_images_list = []
useful_labels_list = []
length_of_dataset_after_filtration = []
acc_list = []

ori_weight_message_string_length_list = []

training_time_list = []
weight_message_string_length = []
classification_time_list = []
append_time_list = []

incremental_training_accuracy_list = []
smote_time_list = []

pruning_time_list = []
pruned_weights_percent_list = []
pruned_weights_list = []
total_nn_weights_list = []

Exemplar_features = 0
Exemplar_labels = 0
first_time = 0
incremental_i = 0
cloud_total_nerons_in_final_layer = starting_softmax_neurons
#cloud_temp_cl = CloudTempClassifier()

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
    global smote_time_list

    global Exemplar_features
    global Exemplar_labels
    global first_time
    global incremental_i
    global acc_list
    global cloud_total_nerons_in_final_layer
    global cloud_cl
    global incremental_training_accuracy_list
    
    global pruning_time_list
    global pruned_weights_percent_list
    global pruned_weights_list
    global total_nn_weights_list

    batch_size = 32
    cloud_criterion = nn.CrossEntropyLoss() #CROSS ENTROPY LOSS FUNCTION
    cloud_learning_rate = 0.0002
    cloud_training_epochs = 30
    cloud_optimizer = optim.Adam(cloud_cl.parameters(), lr=cloud_learning_rate) #ADAM OPTIMIZER

    Feature = Feature.replace(" ","") #remove any spaces
    task_character = Feature[-2] #Check the 2nd last character and decide what the program should do now
    print("task_character: "+str(task_character)+"\n")

    if(task_character == 't'): #train the model

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
                if(len(training_features)<length_of_dataset_after_filtration[incremental_i-1]):
                    total_new_points = length_of_dataset_after_filtration[incremental_i-1] - len(training_features)
                    smote_tic = time.time()
                    remaining_images,remaining_labels = SMOTE(training_features, training_features_labels, total_new_points)
                    smote_toc = time.time()
                    smote_time = smote_toc - smote_tic
                    smote_time_list.append(smote_time)
                    remaining_images = np.reshape(remaining_images,(total_new_points,feature_w,feature_h,feature_d))

                    print("training_features shape: "+str(training_features.shape))
                    print("remaining_images shape: "+str(remaining_images.shape))
                    training_features = np.concatenate((training_features,remaining_images),axis=0)
                    training_features_labels = np.concatenate((training_features_labels,remaining_labels),axis=0)
                    
                    #get initial losses
                    traindataset_new = TensorDataset( Tensor(training_features), Tensor(training_features_labels))
                    Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)

                    print("length of new images: "+str(len(training_features)))
                    length_of_dataset_after_filtration.append(len(training_features))
                    useful_images_list.append(training_features)
                    useful_labels_list.append(training_features_labels)
                else:
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

        pre_training_cl = copy.deepcopy(cloud_cl)
        print("before training, cloud_cl is: "+str(cloud_cl))
        training_time_tic = time.time()
        #Train all the features on the cloud now
        for _ in range(0,cloud_training_epochs):
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
        training_time = (training_time_toc-training_time_tic)
        training_time_list.append(training_time)
        post_training_cl = copy.deepcopy(cloud_cl)

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

    elif(task_character == 'h'):
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

        print("cloud testing append, testing_features_so_far shape: "+str(testing_features_so_far.shape))

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

        print("classification testing_features_so_far shape is: "+str(testing_features_so_far.shape))
        testdatasett = TensorDataset( testing_features_so_far, testing_features_labels_so_far )
        Testloaderr = DataLoader(testdatasett, batch_size= batch_size, shuffle=False)

        correct = 0
        total = 0
        top = 5
        print(cloud_cl)
        with torch.no_grad():
            for data in Testloaderr:
                images, labels = data

                images = images.clone().detach()
                images = images.view(len(images),feature_d,feature_w,feature_h) #RESHAPE THE IMAGES
                labels = labels.long() #MUST CONVERT LABEL TO LONG FORMAT

                images=images.to(device)
                labels=labels.to(device)

                outputs = cloud_cl.forward(images)

                #COMPUTE TOP-5 accuracy
                _,pred=outputs.topk(top,1,largest=True,sorted=True) #select top 5
                total += labels.size(0)

                for j in range(len(pred)):
                    tensor = pred[j]
                    true_label = labels[j]
                    for k in range(len(tensor)):
                        if(true_label == tensor[k].item()):
                            correct += 1
                            break
        if(total>0):
            acc = float(correct/total)*100.0
            print("correct is: "+str(correct)+"\n")
            print("total is: "+str(total)+"\n")
            print("accuracy is: "+str(acc)+"\n")
            acc_list.append(acc)

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

    elif(task_character == 'd'): #means process done
        print("============================CLOUD RESULTS=====================================")
        print("number of classes being learnt at a time: "+str(starting_softmax_neurons))
        print("overall training_time_list: "+str(training_time_list))
        print("TOTAL TRAINING TIME (SECONDS): "+str(sum(training_time_list)))
        print("Testing acc_list: "+str(acc_list))
        print("smote_time_list: "+str(smote_time_list))
        print("TOTAL smote_time_list: "+str(sum(smote_time_list)))
        print("\n\npruned_weights_list: "+str(pruned_weights_list))
        print("TOTAL pruned paramters: "+str(sum(pruned_weights_list)))
        print("total_nn_weights_list: "+str(total_nn_weights_list))
        print("TOTAL paramters: "+str(sum(total_nn_weights_list)))
        print("pruned_weights_percent_list: "+str(pruned_weights_percent_list))
        print("pruning_time_list: "+str(pruning_time_list))
        print("TOTAL pruning_time_list: "+str(sum(pruning_time_list)))
        


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

            outputs = squeezenet.features(images)

            #Convert the tensor to string
            string_tensor = TensorToString(labels,outputs)
            overall_string_tensor = overall_string_tensor + string_tensor + "k"

        #Send the string to the cloud
        overall_string_tensor = overall_string_tensor[:-1] + "c!"
        Cloud(overall_string_tensor)
    

def GetLosses(Trainloader_trail,device,cls):

    special_criterion = nn.CrossEntropyLoss(reduction='none')

    initial_loss_list = []

    image_loss_dictionary = collections.defaultdict(list)
    label_loss_dictionary = collections.defaultdict(list)

    for _, data in enumerate(Trainloader_trail, 0):
        inputs, labels = data

        inputs = inputs.clone().detach()
        inputs = inputs.view(len(inputs),feature_d,feature_w,feature_h)
        labels = labels.long()
                
        inputs=inputs.to(device)
        labels=labels.to(device)

        with torch.no_grad():
            outputs = cls.forward(inputs)
            loss = special_criterion(outputs, labels)
            for j in range(0,len(loss)):
                initial_loss_list.append(loss[j].item())
                image_loss_dictionary[loss[j].item()].append(inputs[j])
                label_loss_dictionary[loss[j].item()].append(labels[j].cpu().numpy())

    return initial_loss_list, image_loss_dictionary, label_loss_dictionary

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
old_mean_loss = 0
def Cutoff(mean,median,standard_deviation,image_loss_dictionary,label_loss_dictionary,depth,width,height):
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
    
    omega = 1.0
    number_of_samples_to_be_dropped  = int(omega * number_of_samples_to_be_dropped)
    
    print("number_of_samples_to_be_dropped: "+str(number_of_samples_to_be_dropped))

    #get all the images and labels above the cutoff value
    number_of_samples_to_selected = 0
    new_images = 0
    new_labels = []
    filtered_original_losses = []

    #"""
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
    
    
    #RANDOM SAMPLING
    for i in range(0,len(randomly_selected_losses)):
        key_loss = randomly_selected_losses[i]
        #if(key_loss not in randomly_selected_losses):
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

    return new_images,new_labels#,filtered_original_losses

#MAIN FUNCTION FOR PERFORMING DATA SAMPLING
def GetusefulData(images_train,labels_train,batch_size,cls):

    traindataset = TensorDataset( images_train, Tensor(labels_train) )
    Trainloader_trail = DataLoader(traindataset, batch_size= batch_size, shuffle=False)

    #Do single forward pass and get the losses
    initial_loss_list, image_loss_dictionary, label_loss_dictionary = GetLosses(Trainloader_trail,device,cls)
    sorted_initial_list = sorted(initial_loss_list)
    mean, median, standard_deviation = GetLossStatistics(sorted_initial_list)
    new_images, new_labels = Cutoff(mean,median,standard_deviation,image_loss_dictionary,label_loss_dictionary,feature_d,feature_w,feature_h)
    

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

def DataSampling(per_class_features, per_class_labels_list_np_array, Batch_size, edge_cl, edge_temp_cl,number_of_classes_to_be_added_ever_increment,incremental_i):

    new_images = 0
    print("per class data sampling begins here \n")
    per_class_labels_list_np_array = np.asarray(per_class_labels_list)
    if(incremental_i == 0):
        edge_cl.eval()
        new_images, new_labels = GetusefulData(per_class_features, per_class_labels_list_np_array, Batch_size, edge_cl)
        edge_cl.train()
    else:
        for kk in range(0,len(per_class_labels_list)):
            per_class_labels_list[kk] -= (number_of_classes_to_be_added_ever_increment*incremental_i)
        per_class_labels_list_np_array = np.asarray(per_class_labels_list)
        edge_temp_cl.eval()
        new_images, new_labels = GetusefulData(per_class_features, per_class_labels_list_np_array, Batch_size, edge_temp_cl)
        edge_temp_cl.train()
        for kk in range(0,len(new_labels)):
            new_labels[kk] += (number_of_classes_to_be_added_ever_increment*incremental_i)
    
    return new_images, new_labels

#==========MAIN PROGRAM==========
squeezenet = models.mobilenet_v2(pretrained=True)
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


training_image_list = 0
training_image_label_list = []
testing_image_list = 0
testing_image_label_list = []
#must use the absolute path
directory = '/home/swarajdubey611/Stanford_dogs'#r"C:\Users\Swaraj\AppData\Local\Programs\Python\Python36\Journal_paper_codes\SqueezeNet\CUB_200_2011\images"
#directory = '/content/gdrive/My Drive/CUB_200_resized'
index_counter = 0
train_counter = 0
test_counter = 0
class_counter = -1
Exemplar_features = 0
Exemplar_labels = 0
classes_per_training_round_count = 0
edge_incremental_i = 0
number_of_classes_to_be_added_ever_increment = starting_softmax_neurons
total_nerons_in_final_layer= starting_softmax_neurons

#Variables for SMOTE
useful_images_list = []
useful_labels_list = []
length_of_dataset_after_filtration = []

Batch_size = 32
criterion = nn.CrossEntropyLoss() #CROSS ENTROPY LOSS FUNCTION
learning_rate = 1e-4
optimizer = optim.Adam(edge_cl.parameters(), lr=learning_rate) #ADAM OPTIMIZER
total_epochs = 15
batch_size = 128 #this batch size is only for data loading

training_features_list = 0
testing_features_list = 0
training_features_list_labels = []
testing_features_list_labels = []
folder_count = 0
train_batch_counter = 0
test_batch_counter = 0
all_testing_features_so_far = 0
all_testing_labels_so_far = []
edge_processing_time_list = []

samples_transmitted_count = 0
length_so_far = 0
number_of_overloads = 0
datasize_sent_list = []
overall_string_tensor = ""
testing_overall_string_tensor = ""
total_append_operation_time = 0

new_images_length_counter = 0
new_images_length_list = []

train_mat = loadmat('/home/swarajdubey611/train_list.mat')
test_mat = loadmat('/home/swarajdubey611/test_list.mat')

training_files = train_mat['file_list']
testing_files = test_mat['file_list']

total_classes = 120
img_dimension = 85
overall_tic = time.time()
for subdir in os.listdir(directory):
    if(edge_incremental_i < total_classes):# (classes_per_training_round_count < number_of_classes_to_be_added_ever_increment) and (incremental_i < number_of_increments)
        print(subdir)
        subdir_absolute_path = directory + '/' + subdir
        class_counter += 1
        folder_count += 1
        
        for f in os.listdir(subdir_absolute_path):
            file_name = subdir + '/' + f
            ori_img = Image.open(os.path.join(subdir_absolute_path,f)) #this is the JPEG image
            width, height = ori_img.size  
            newsize = (img_dimension, img_dimension) 
            ori_img = ori_img.resize(newsize) 
            img = transform(ori_img)
            if(img.shape[0] == 3):#accept only RGB images
                img = img.reshape((1,3,img_dimension,img_dimension)) #reshape the image tensor
                
                try: #this means the file is in the training set
                    result = np.where(training_files == file_name)[0][0]
                    if(train_counter == 0): #first time appending
                        training_image_list = img
                        train_counter = 1
                    else:
                        training_image_list = np.concatenate((training_image_list,img),axis=0)
                    training_image_label_list.append(class_counter)

                except: #this means the file is in the testing set
                    if(test_counter == 0): #first time appending
                        testing_image_list = img
                        test_counter = 1
                    else:
                        testing_image_list = np.concatenate((testing_image_list,img),axis=0)
                    testing_image_label_list.append(class_counter)
            
            
        if(folder_count == 1): #Send append message to the cloud
            print("Training images size: "+str(len(training_image_label_list)))
            print("Testing images size: "+str(len(testing_image_label_list)))
            print("Starting data sampling")
            folder_count = 0
            edge_tic = time.time()
            #OBTAIN FEATURES OF ALL SAMPLES IN THE SUB FOLDER
            trail_train_dataset = TensorDataset( Tensor(training_image_list), Tensor(np.asarray(training_image_label_list)) )
            Trialtrainloader = DataLoader(trail_train_dataset, batch_size= batch_size, shuffle=False)
            training_image_list = 0
            training_image_label_list = []

            per_class_features = 0
            per_class_features_counter = 0
            per_class_labels_list = []
            for i, data in enumerate(Trialtrainloader, 0):
                inputs, _ = data

                inputs = inputs.clone().detach()
                inputs = inputs.view(len(inputs),3,img_dimension,img_dimension) 
                inputs=inputs.to(device)

                for _ in range(len(inputs)):
                    per_class_labels_list.append(class_counter)

                with torch.no_grad():
                    outputs = squeezenet.features(inputs)
                    print("outputs size: "+str(outputs.shape))
                    if(per_class_features_counter==0):
                        per_class_features = outputs
                        per_class_features_counter = 1
                    else:
                        per_class_features = torch.cat([per_class_features,outputs],dim=0)


            #Carry out data sampling here
            per_class_labels_list_np_array = np.asarray(per_class_labels_list)
            new_images, new_labels = DataSampling(per_class_features, per_class_labels_list_np_array, Batch_size, edge_cl, edge_temp_cl,number_of_classes_to_be_added_ever_increment,edge_incremental_i)
            
            new_images_length_counter += len(new_images)
            print("per class data sampling ends here \n")
            Trainloader_iterations = 0
            if(len(new_images) % batch_size == 0):
                Trainloader_iterations = len(new_images) // batch_size
            else:
                Trainloader_iterations = (len(new_images) // batch_size) + 1
            edge_toc = time.time()
            edge_processing_time_list.append(edge_toc-edge_tic)

            samples_transmitted_count += len(new_images)
            
            #Convert the filtered data samples to string format
            traindataset_new = TensorDataset( Tensor(new_images), Tensor(new_labels))
            Trainloader = DataLoader(traindataset_new, batch_size= batch_size, shuffle=True)

            ti = 0
            for i, data in enumerate(Trainloader, 0): #This will only run once i.e. only 1 epoch        
                mini_batch_tic = time.time()
                inputs, labels = data

                inputs = inputs.clone().detach()
                inputs = inputs.view(len(inputs),feature_d,feature_w,feature_h)
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
                    ti = time_diff
                else:
                    if((time_diff - ti) > 0.5):
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


            #get testing features of images and only save that
            ti = 0
            trail_test_dataset = TensorDataset( Tensor(testing_image_list), Tensor(np.asarray(testing_image_label_list)) )
            Trialtestloader = DataLoader(trail_test_dataset, batch_size= batch_size, shuffle=False)
            testing_image_list = 0
            testing_image_label_list = []
            for i, data in enumerate(Trialtestloader, 0):
                mini_batch_tic = time.time()

                inputs, labels = data

                inputs = inputs.clone().detach()
                inputs = inputs.view(len(inputs),3,img_dimension,img_dimension)
                labels = labels.long()

                inputs=inputs.to(device)
                labels=labels.to(device)

                with torch.no_grad():
                    outputs = squeezenet.features(inputs)
                string_tensor = TensorToString(labels,outputs)
                testing_overall_string_tensor = testing_overall_string_tensor + string_tensor + "k"
                mini_batch_toc = time.time()
                time_diff = mini_batch_toc - mini_batch_tic
                if(i==0):
                    ti = time_diff
                else:
                    if((time_diff - ti) > 0.5):
                        if(classes_per_training_round_count == number_of_classes_to_be_added_ever_increment-1):
                            testing_overall_string_tensor = testing_overall_string_tensor[:-1] + "c!"
                            Cloud(testing_overall_string_tensor)
                        else:
                            testing_overall_string_tensor = testing_overall_string_tensor[:-1] + "h!"
                            Cloud(testing_overall_string_tensor)
                        testing_overall_string_tensor = ""

            #reset all the variables
            training_image_list = 0
            testing_image_list = 0
            train_counter = 0
            test_counter = 0
            training_image_label_list = []
            testing_image_label_list = []


        if(classes_per_training_round_count == number_of_classes_to_be_added_ever_increment-1): #Send to the cloud for training

            new_images_length_list.append(samples_transmitted_count)
            samples_transmitted_count = 0
            print("before training edge_incremental_i: "+str(edge_incremental_i))
            if(len(overall_string_tensor)>1):
                overall_string_tensor = overall_string_tensor[:-1] + "t!" #remove the last k and add t! at the end
                length_so_far += len(overall_string_tensor)
                length_so_far *= 1e-6
                datasize_sent_list.append(length_so_far)
                Cloud(overall_string_tensor)        
            overall_string_tensor = ""            


            #Start the classification process
            if(len(testing_overall_string_tensor)>1):
                testing_overall_string_tensor = testing_overall_string_tensor[:-1] + "c!"
                Cloud(testing_overall_string_tensor)
            testing_overall_string_tensor = ""

            #Modify the edge_cl
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

            #reset all the variables for the next incremental training round
            training_features_list = 0
            training_features_list_labels = []
            testing_features_list = 0
            testing_features_list_labels = []
            train_batch_counter = 0
            test_batch_counter = 0
            edge_incremental_i += 1
            classes_per_training_round_count = -1

            print("after training edge_incremental_i: "+str(edge_incremental_i))
        else: #Send to the cloud as append append
            print("acloud append")
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

            if(len(testing_overall_string_tensor)>1):
                testing_overall_string_tensor = testing_overall_string_tensor[:-1] + "h!"
                Cloud(testing_overall_string_tensor)
            testing_overall_string_tensor = ""

    classes_per_training_round_count += 1

stop_cloud_message = "d!"
Cloud(stop_cloud_message)
overall_toc = time.time()
print("=========================EDGE RESULTS===========================")
print("overall time: "+str(overall_toc - overall_tic))
print("new_images_length_list: "+str(new_images_length_list))
print("total images sent: "+str(sum(new_images_length_list)))
print("amount of data sent so far (MB): ")
print(datasize_sent_list)
print("TOTAL data sent (MB): ")
print(sum(datasize_sent_list))
