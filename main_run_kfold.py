import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import copy
import sys
import configparser
import numpy as np

from torchsummary import summary

from sklearn.model_selection import KFold

from utils import load_xca_dataset
from train import train_model
from test import test_model, eval_preds
from models import resnets
from gradcam.GradCam import GradCam, computeGradCam



size_output_layer4 = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048}


def unfreeze_model(net):
    for param in net.parameters():
        param.requires_grad = True

    return net

def load_weights(net, model_checkpoint, imagenet, device):
    checkpoint = torch.load(model_checkpoint, map_location=torch.device(device))
    if imagenet:
        print('Loading weights from imagenet')
        net.load_state_dict(checkpoint,)
    else:
        print('Loading weights from model_state_dict')
        net.load_state_dict(checkpoint['model_state_dict'])

    return net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# read parameters
args = str(sys.argv)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# read configuration parameters
config = configparser.ConfigParser()
config.read(sys.argv[1])

DATA_DIR = config.get('PARAMS', 'DATA_DIR')
WEIGHTS_DIR = config.get('PARAMS', 'WEIGHTS_DIR')
RESULTS_DIR = config.get('PARAMS', 'RESULTS_DIR')
torch_seed = int(config.get('PARAMS', 'torch_seed'))
model_name = config.get('PARAMS', 'model_name')
model_depth = int(config.get('PARAMS', 'model_depth'))
lr = float(config.get('PARAMS', 'lr'))
momentum = float(config.get('PARAMS', 'momentum'))
factor = float(config.get('PARAMS', 'factor'))
patience = int(config.get('PARAMS', 'patience'))
batch_size = int(config.get('PARAMS', 'batch_size'))
num_epochs = int(config.get('PARAMS', 'num_epochs'))
model_checkpoint = config.get('PARAMS', 'model_checkpoint')
finetuning = True if config.get('PARAMS', 'finetuning') == 'True' else False
run_test = True if config.get('PARAMS', 'run_test') == 'True' else False
CBAM = True if config.get('PARAMS', 'CBAM') == 'True' else False
GRADCAM = True if config.get('PARAMS', 'GRADCAM') == 'True' else False
imagenet_norm = True if config.get('PARAMS', 'imagenet_norm') == 'True' else False
imagenet_init = True if config.get('PARAMS', 'imagenet_init') == 'True' else False
from_scratch = True if config.get('PARAMS', 'from_scratch') == 'True' else False

# set manual seed
torch.manual_seed(torch_seed)

# create model

print('--> Create ResNet model')
resnet_name = 'resnet%d' % model_depth
#create model
if CBAM:
    original_net = resnets.create_model(resnet_name, 'cbam', rd_ration=1)
else:
    original_net = resnets.create_model(resnet_name, 'None')


#summary(original_net, (3, 32, 32))
if not from_scratch:
    if imagenet_init:
        print('--> ImageNet weights')
        original_net = load_weights(original_net, model_checkpoint, imagenet_init, device)
        original_net.fc = torch.nn.Linear(size_output_layer4[resnet_name], 1)
    else:
        if not CBAM:
            original_net.fc = torch.nn.Linear(size_output_layer4[resnet_name], 1)
        print('--> Finetuning weights')
        original_net = load_weights(original_net, model_checkpoint, imagenet_norm, device)
else:
    print('--> Scratch weights')
    if not CBAM:
        original_net.fc = torch.nn.Linear(size_output_layer4[resnet_name], 1)
    #if imagenet_norm:
        #original_net.fc = torch.nn.Linear(size_output_layer4[resnet_name], 1)

original_net.to(device)

# create k-fold
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=torch_seed)

all_accuracy = np.zeros(k_folds)
all_precision = np.zeros(k_folds)
all_recall = np.zeros(k_folds)
all_f1 = np.zeros(k_folds)
all_specificity = np.zeros(k_folds)

# load & merge train/val
# load dataset
dataset = load_xca_dataset.load_kfold_dataset(DATA_DIR, imagenet=imagenet_norm)

#summary(original_net, (3, 32, 32))

#check if WEIGHTS_DIR exists
try:
    os.makedirs(WEIGHTS_DIR)
    print("Directory '%s' created successfully" % WEIGHTS_DIR)
except OSError as error:
    print("Directory '%s' can not be created or allready exist" % WEIGHTS_DIR)

#check if RESULTS_DIR exists
try:
    os.makedirs(RESULTS_DIR)
    print("Directory '%s' created successfully" % RESULTS_DIR)
except OSError as error:
    print("Directory '%s' can not be created or allready exist" % RESULTS_DIR)
    

for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(dataset)))):
    print(f'FOLD {fold}')
    print('-' * 50)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    # merge train & val data loaders
    dataloaders = {'train': trainloader, 'validation': valloader}
    dataset_sizes = {'train': len(train_ids), 'validation': len(test_ids)}

    net = copy.deepcopy(original_net)
    net.to(device)

    # print model summary
    #summary(net, (3, 32, 32))

    # create optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=factor, patience=patience,
                                               verbose=True)

    # train the model
    model_checkpoint = '%s/model_%s_fold_%d.pth' % (WEIGHTS_DIR, model_name, fold)
    model_history = '%s/history_%s_fold_%d.json' % (WEIGHTS_DIR, model_name, fold)

    print('-' * 50)
    net, m_history = train_model(device=device, model=net, criterion=criterion,
                                   optimizer=optimizer, scheduler=scheduler,
                                   num_epochs=num_epochs, batch_size=batch_size,
                                   dataloaders=dataloaders,
                                   dataset_sizes=dataset_sizes,
                                   PATH_MODEL=model_checkpoint,
                                   PATH_HISTORY=model_history)

    if run_test:
        # test the model
        results_report = '%s/model_%s_fold_%d.json' % (RESULTS_DIR, model_name, fold)
        results_report_probas = '%s/result_probas_model_%s_fold_%d.csv' % (RESULTS_DIR, model_name, fold)

        dataloaders_test, dataset_test_sizes = load_xca_dataset.load_test_dataset(DATA_DIR,
                                                                                  imagenet=imagenet_norm)

        print('-' * 50)
        print('Testing step')
        # unfreeze model to apply gradcam
        net = unfreeze_model(net)

        # set model to eval
        net.eval()
        y_true, y_pred = test_model(device=device, model=net, test_loader=dataloaders_test['test'],
                                    PATH_RESULTS=results_report_probas)
        results = eval_preds(y_true, y_pred, results_report)

        if GRADCAM:
            #if imagenet_init: # baseline model
            #    gradcam_layer = 'layer4'
            #else:
            gradcam_layer = 'layer3'
            gradcam_name = '%s_fold_%d'%(model_name, fold)
            gradcam = GradCam(model=net, cam_layer_name=gradcam_layer, imagenet_norm=imagenet_norm)
            computeGradCam(gradcam, dataloaders_test['test'], device, RESULTS_DIR, gradcam_name)

        all_accuracy[fold] = results['accuracy']
        all_precision[fold] = results['precision']
        all_recall[fold] = results['recal']
        all_f1[fold] = results['f1-score']
        all_specificity[fold] = results['specificity']

# get statistic from results
print("ACC: %.4f (+/- %.4f)" % (np.mean(all_accuracy), np.std(all_accuracy)))
print("PC: %.4f (+/- %.4f)" % (np.mean(all_precision), np.std(all_precision)))
print("REC: %.4f (+/- %.4f)" % (np.mean(all_recall), np.std(all_recall)))
print("F1: %.4f (+/- %.4f)" % (np.mean(all_f1), np.std(all_f1)))
print("SP: %.4f (+/- %.4f)" % (np.mean(all_specificity), np.std(all_specificity)))
