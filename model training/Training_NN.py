import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import utils.models as models
#from Create_Data import DataCreator
from sklearn.model_selection import train_test_split

# Class Trainer for training a Neural Network
class Trainer():

    models = {
        "1": models.Model_1,
        "2": models.Model_2,
        "3": models.Model_3,
        "4": models.Model_4 # Hier muss für neue Modelle die entsprechende Zeile eingefügt werden.
        #"5": models.Model_5,
    }

    '''Sonstige torch-Objekte:'''
    sm = nn.Softmax(-1)
    criterion_cl = nn.CrossEntropyLoss()
    criterion_re = nn.MSELoss()
    optimizer = optim.Adam

    def __init__(self, train_split = 0.8, lr = 0.001, model_number="4", model_name = None, dataset_name = None, epochs=10, batch_size=10, nn_type = "classification"):
        '''
        :param lr: Learning Rate
        :param model_number: Welches der Modelle von self.models?
        :param model_name: Speichername des Modells (falls None, wird dieser automatisch bestimmt)
        :param epochs: Anzahl der Trainingsepochen mit normalem Training
        :param batch_size: Größe der Batch-Size beim normalen Training
        :param nn_type: Art des Modells (Klassfikation oder Regression)
        '''

        sm = nn.Softmax(-1)
        criterion_cl = nn.CrossEntropyLoss()
        criterion_re = nn.MSELoss()
        optimizer = optim.Adam

    
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_split = train_split
        self.nn_type = nn_type
        self.model = self.models[model_number]()
        if model_name == None:
            self.model_name = "model_" + model_number + "_" + str(epochs)
        else:
            self.model_name = model_name
        # self.folder = os.path.join("saves", self.model_name)
        self.dataset_name = dataset_name
        self.folder = f'..\datasets\{self.dataset_name}'
        self.location = os.path.join(self.folder, f"{self.model_name}.pt")
        self.test_data = None

        if self.nn_type == "classification":
            self.criterion = criterion_cl
        elif self.nn_type == "regression":
            self.criterion = criterion_re

    def initialize_data(self, data = None, labels = None, train_data = None, test_data = None):
        '''
        Erstellung der Daten und deren zugehörigen Label mithilfe einer SampleMethod-Instanz
        '''
      
        if (train_data is None) or (test_data is None): 
            self.data = data
            self.labels = labels
        else:
            self.data = train_data[0]
            self.labels = train_data[1]
            self.test_data = test_data[0]
            self.test_labels = test_data[1]

            
         
    def prepare_for_training(self):
        '''
        Vorbereitung der torch Dataloader für Training und Test
        '''

        def split_train_test(l):
            return l[-self.num_test:], l[:-self.num_test]

        self.data = torch.FloatTensor(self.data)
        if self.nn_type == "classification":
            self.labels = torch.LongTensor(self.labels)
        elif self.nn_type == "regression":
            self.labels = torch.FloatTensor(self.labels)
        
        if self.test_data is None:
            self.data, self.test_data, self.labels, self.test_labels = train_test_split(self.data, self.labels, test_size = 1 - self.train_split, random_state = 42)
        else:
            self.test_data  = torch.FloatTensor(self.test_data)
            if self.nn_type == "classification":
                self.test_labels = torch.LongTensor(self.test_labels)
            elif self.nn_type == "regression":
                self.test_labels = torch.FloatTensor(self.test_labels)
           
        self.train_dataset = TensorDataset(self.data, self.labels)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=16)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=True, batch_size=1)

    def save_model(self, model):
        '''
        Speichert die Modellparameter für den späteren Gebrauch mit Propagationmethoden (vgl. Klasse Model in PropagationBase.py)
        Oder um zwischen self.model und self.adf_model Parameter zu übertragen
        :param model: Entweder self.model oder self.adf_model
        '''
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        param_dict = {}
        for i in range(self.model.num_layers):
            param_dict["weights_" + str(i)] = model.linear_layers[i].weight.data
            param_dict["bias_" + str(i)] = model.linear_layers[i].bias.data
        torch.save(param_dict, self.location)
            
    def load_model(self, model):
        '''
        vgl. self.save_model()
        '''
        param_dict = torch.load(self.location)
        for i in range(self.model.num_layers):
            model.linear_layers[i].weight.data = param_dict["weights_" + str(i)]
            model.linear_layers[i].bias.data = param_dict["bias_" + str(i)]

    def train_one_epoch(self, adf=False):
        '''
        Training von self.model  um eine Epoche
        :param adf: True <--> self.adf_model wird trainiert
        '''
      
        model = self.model
        data_loader = self.train_dataloader
        criterion = self.criterion
        
        optimizer = self.optimizer(params=model.parameters(), lr = self.lr)
        for i, batch in enumerate(data_loader):
            model.zero_grad()
            if adf:
                input = batch[0], batch[2]
            else:
                input = [batch[0]]
            output = model(*input)
            #b1 = batch[1].reshape((batch[1].size()[0], 1))
            b1 = batch[1]
            loss = criterion(output, b1)
            loss.backward()
            optimizer.step()

    def test_model(self, adf=False):
        '''
        Analog zu self.train_one_epoch()
        '''
        
        model = self.model
        data_loader = self.test_dataloader
        data_loader_train = self.train_dataloader
        print(len(self.test_dataloader))
        
        if self.nn_type == "classification":
            counter = 0
            for i, batch in enumerate(data_loader):
                model.zero_grad()
                if adf:
                    input = batch[0], batch[2]
                else:
                    input = [batch[0]]
                output = model(*input)
                smout = self.sm(output)
                if adf:
                    smout = output / output.sum(dim=-1, keepdim=True)
                for i in range(len(smout)):
                    if torch.argmax(smout[i].data) == batch[1][i]:
                        counter += 1
                    else:
                        #pass
                        print("Fehler Test:", smout[i].data, batch[1][i])
 
            test_acc = counter/len(self.test_labels)
                
            '''accuracy on training data (ToDo: nimmt noch ganze Batches anstatt einzelner Instanzen)'''
            counter = 0
            for i, batch in enumerate(data_loader_train):
                model.zero_grad()
                if adf:
                    input = batch[0], batch[2]
                else:
                    input = [batch[0]]
                output = model(*input)
                smout = self.sm(output)
                if adf:
                    smout = output / output.sum(dim=-1, keepdim=True)
                for i in range(len(smout)):
                    if torch.argmax(smout[i].data) == batch[1][i]:
                        counter += 1
                    else:
                        pass
                        #print("Fehler Training:", smout[i].data, batch[1][i])
            
            train_acc = counter/len(self.labels) # Falls man die Trainings Accuracy wissen will, diese Variable auch mit ausgeben ! 
            
            return test_acc #,train_acc

        elif self.nn_type == "regression":
            mse_list = np.zeros(len(self.test_dataloader))
            for i, batch in enumerate(data_loader):
                input = [batch[0]]
                output = model(*input)
                b1 = batch[1].reshape((batch[1].size()[0], 1))
                mse_list[i] = self.criterion(output, b1)
            return np.mean(mse_list)

    def train_and_save(self):
        '''
        Training für alle Epochen und ADF-Epochen
        Speichert das trainierte Modell
        '''
        for i in range(self.epochs):
            self.train_one_epoch()
        self.save_model(self.model)

    def test(self):
        '''
        Test des vollständig trainierten Modells
        '''
        return self.test_model()
    
        
if __name__ == "__main__" and 0:
    trainer = Trainer(lr=0.001, epochs=500, model_number="8", model_name=None)
    trainer.initialize_data()
    trainer.prepare_for_training()
    trainer.train_and_save()
    print("Accuracy bzw. MSE auf Testdaten:", trainer.test())
