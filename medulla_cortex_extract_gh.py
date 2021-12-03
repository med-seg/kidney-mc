import sys
import os
from os import path

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd

import matplotlib
# matplotlib.axes.Axes.plot
# matplotlib.pyplot.plot
# matplotlib.axes.Axes.legend
# matplotlib.pyplot.legend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

import scipy
from scipy.ndimage import zoom
from scipy import signal
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from scipy import ndimage

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

from skimage import morphology
from skimage import measure
from skimage import io, filters
from skimage import data
from skimage.filters import threshold_otsu
from skimage import data, color, io, img_as_float

from sklearn import cluster
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

import cv2
import math
from PIL import Image, ImageEnhance 

import nibabel as nib
import pickle
import funcs_mc

def computeNew4DAfterBasline(im, baseline):    
    
    vol4D = np.array(im);
    xl = vol4D.shape[0];
    yl = vol4D.shape[1];
    sl = vol4D.shape[2];
    #tl = vol4D.shape[3];
    
    baselineIntensities = np.array(vol4D[:,:,:,1:baseline-1]);
    reshapeBaseline =  np.reshape(baselineIntensities,[xl*yl*sl,baselineIntensities.shape[3]]);
    fid =  np.mean(reshapeBaseline,axis=0);
    maxMean = max(fid);
    
    newVol4D = np.copy(vol4D);
    oldVol4D = np.copy(vol4D);
    
    newVol4D[:,:,:,baseline:] = np.subtract(oldVol4D[:,:,:,baseline:],maxMean)
    #newVol4D[:,:,:,:] = np.subtract(oldVol4D[:,:,:,:],maxMean)
    
    return newVol4D

def computeNew4D(im):    
    
    vol4D = np.array(im);
    xl = vol4D.shape[0];
    yl = vol4D.shape[1];
    sl = vol4D.shape[2];
    tl = vol4D.shape[3];
    
    reshapedVol4D = np.reshape(vol4D,[xl*yl*sl,tl]);
    fid =  np.mean(reshapedVol4D,axis=0)
    
    diffFid = np.diff(fid,n=1,axis=0) 
    absDiffFid = np.absolute(diffFid)
        
    maxAbsDiffFid = max(absDiffFid)
    
    jumpTimeSample, = np.where(absDiffFid == maxAbsDiffFid)
    jumpTimeSample = max(jumpTimeSample)
    
#    check1 = fid[jumpTimeSample-1]   
#    check2 = fid[jumpTimeSample]   
#    check3 = fid[jumpTimeSample+1]     
#    check4 = vol4D[:,:,:,jumpTimeSample+1:];  
    
    difference = fid[jumpTimeSample+1]-fid[jumpTimeSample];
    
    newVol4D = np.copy(vol4D);
    oldVol4D = np.copy(vol4D);

#    before = vol4D[:,:,1,jumpTimeSample:jumpTimeSample+1]
#    mB = np.mean(before);
#    after  = vol4D[:,:,1,jumpTimeSample:jumpTimeSample+1]-difference; 
#    mA = np.mean(after);
   
    newVol4D[:,:,:,jumpTimeSample+1:] = np.subtract(oldVol4D[:,:,:,jumpTimeSample+1:],(difference*2))
    
    return newVol4D,difference

    
def processSingleSubjects_ha(patientName,subjectInfo,baselineTime):    
    
    reconMethod='SCAN';
    
    numPC=5;
    pca = PCA(n_components=numPC)
    
    dx=64;dy=64;dz=32;
     
    print(patientName)
    
    vol4D00, KM, Box, rkmOri, lkmOri = funcs_mc.readData4(patientName,subjectInfo,reconMethod,1);
    
    copyKM = np.copy(KM);
    BoxCopy = np.copy(Box)
    #numSlicesVol = vol4D00.shape[2];
    
    # timeRes0=subjectInfo['timeRes'][patientName];
    timeRes00 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']     
    index = timeRes00.index.values
    index = index[0]
    timeRes0 = timeRes00.loc[index]
    # timeRes0 = timeRes00[0]  

    if not isinstance(timeRes0, (int, float)):
        timeRes=float(timeRes0.split("[")[1].split(",")[0]);
    else:
        timeRes=np.copy(timeRes0); 
        
        
    im = np.copy(vol4D00);
    medianFind = np.median(im);
    if medianFind == 0:
        medianFind = 1.0;
         
    im=im[:,:,:,baselineTime:];
    im=im/medianFind;
    
    vol4D0 = im.copy();
    copyVol4D0 = vol4D0
    
    origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
    resamTimeResVec=np.arange(0,50*6,6);   # resample to 50 data point
    
    if origTimeResVec[-1]<resamTimeResVec[-1]:
        print(patientName)
    
    Box=Box.astype(int)
    kidneyNone=np.nonzero(np.sum(Box,axis=1)==0); #right/left
    if kidneyNone[0].size!=0:
        kidneyNone=np.nonzero(np.sum(Box,axis=1)==0)[0][0]; #right/left
    
    KM[KM>1]=1;
    if Box[0,2]+Box[0,5]+3 >= KM.shape[2] or Box[0,2]+Box[0,5]-3 <0:
        Box[:,[3,4,5]]=Box[:,[3,4,5]]+[10,10,0];
    else:
        Box[:,[3,4,5]]=Box[:,[3,4,5]]+[10,10,3];
    
    # Box[:,[3,4,5]]=Box[:,[3,4,5]]+[15,15,3];         
    vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
    PCs=pca.fit_transform(vol4Dvecs)
    vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC))

    
    if kidneyNone!=0:
        croppedData4DR_pcs=vol4Dpcs[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
        croppedData4DR=vol4D0[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
        KMR=KM[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2)];  
        
        croppedData4DR_pcs=zoom(croppedData4DR_pcs,(dx/np.size(croppedData4DR_pcs,0),dy/np.size(croppedData4DR_pcs,1),dz/np.size(croppedData4DR_pcs,2),1),order=0);
        croppedData4DR=zoom(croppedData4DR,(dx/np.size(croppedData4DR,0),dy/np.size(croppedData4DR,1),dz/np.size(croppedData4DR,2),1),order=0);

        f_out = interp1d(origTimeResVec,croppedData4DR, axis=3,bounds_error=False,fill_value=0)     
        croppedData4DR = f_out(resamTimeResVec);  

        KMR=zoom(KMR,(dx/np.size(KMR,0),dy/np.size(KMR,1),dz/np.size(KMR,2)),order=0);
        
#       croppedData4DR_pcs[KMR<=0] = 0;
#       croppedData4DR[KMR<=0] = 0;
        

    if kidneyNone!=1:    
        croppedData4DL_pcs=vol4Dpcs[Box[1,0]-int(Box[1,3]/2):Box[1,0]+int(Box[1,3]/2),\
                                Box[1,1]-int(Box[1,4]/2)+10:Box[1,1]+int(Box[1,4]/2)-10,\
                                Box[1,2]-int(Box[1,5]/2):Box[1,2]+int(Box[1,5]/2),:];    
        croppedData4DL=vol4D0[Box[1,0]-int(Box[1,3]/2):Box[1,0]+int(Box[1,3]/2),\
                                Box[1,1]-int(Box[1,4]/2)+10:Box[1,1]+int(Box[1,4]/2)-10,\
                                Box[1,2]-int(Box[1,5]/2):Box[1,2]+int(Box[1,5]/2),:];    
        KML=KM[Box[1,0]-int(Box[1,3]/2):Box[1,0]+int(Box[1,3]/2),\
                                Box[1,1]-int(Box[1,4]/2)+10:Box[1,1]+int(Box[1,4]/2)-10,\
                                Box[1,2]-int(Box[1,5]/2):Box[1,2]+int(Box[1,5]/2)];    


        croppedData4DL_pcs=zoom(croppedData4DL_pcs,(dx/np.size(croppedData4DL_pcs,0),dy/np.size(croppedData4DL_pcs,1),dz/np.size(croppedData4DL_pcs,2),1),order=0);
        croppedData4DL=zoom(croppedData4DL,(dx/np.size(croppedData4DL,0),dy/np.size(croppedData4DL,1),dz/np.size(croppedData4DL,2),1),order=0);

        f_out = interp1d(origTimeResVec,croppedData4DL, axis=3,bounds_error=False,fill_value=0)     
        croppedData4DL = f_out(resamTimeResVec); 
        
        KML=zoom(KML,(dx/np.size(KML,0),dy/np.size(KML,1),dz/np.size(KML,2)),order=0);
        
#       croppedData4DL_pcs[KML<=0] = 0;
#       croppedData4DL[KML<=0] = 0;

    if kidneyNone==0:
        print('No right kidney')
        croppedData4DR = [];
        croppedData4DR_pcs = [];
        KMR = [];
        d=np.concatenate((croppedData4DL[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DL_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
        #KM2=np.concatenate((KML[np.newaxis,:,:,:],KML[np.newaxis,:,:,:]),axis=0);   

    elif kidneyNone==1:
        print('No left kidney')
        croppedData4DL = [];
        croppedData4DL_pcs = [];
        KML = [];
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DR[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DR_pcs[np.newaxis,:,:,:,:]),axis=0);
        #KM2=np.concatenate((KMR[np.newaxis,:,:,:],KMR[np.newaxis,:,:,:]),axis=0);          
    else:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
        #KM2=np.concatenate((KMR[np.newaxis,:,:,:],KML[np.newaxis,:,:,:]),axis=0);        
        
    #d[d<0]=0;
    d = d/d.max()
    dpcs = dpcs/dpcs.max()
    Box = Box/Box.max()
    
    f_out = interp1d(origTimeResVec,copyVol4D0, axis=3,bounds_error=False,fill_value=0)     
    copyVol4D0 = f_out(resamTimeResVec);       
    
    vol4Dvecs=np.reshape(copyVol4D0, (copyVol4D0.shape[0]*copyVol4D0.shape[1]*copyVol4D0.shape[2], copyVol4D0.shape[3]));
    PCs=pca.fit_transform(vol4Dvecs)
    #copyVol4Dpcs=np.reshape(PCs, (copyVol4D0.shape[0],copyVol4D0.shape[1],copyVol4D0.shape[2], numPC))

    #da0_pcs=copyVol4Dpcs.T/copyVol4Dpcs.max();
    #da0=copyVol4D0.T/copyVol4D0.max();
    
    
    return  kidneyNone, croppedData4DL, croppedData4DR, croppedData4DL_pcs, croppedData4DR_pcs, KML,KMR, copyVol4D0, rkmOri, lkmOri, BoxCopy, copyKM      
    #processSingleSubjects_ha(patientName,subjectInfo,baselineTime):  

def to_rgb3a(im):
    return np.dstack([im.astype(np.uint8)] * 3)

def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def to_binary(img, lower, upper):
    return (lower < img) & (img < upper)

# def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.3):
    
#     out = img.copy()
#     img_layer = img.copy()
#     img_layer[mask] = color
#     out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
#     return(out)

def save_image(array, name):

    array = array*255.
    fig = plt.figure()
    plt.imsave(name, array.astype('uint8'), cmap=matplotlib.cm.gray, vmin=0, vmax=255)
    plt.close(fig)
    

# begin medulla and cortex segmentation

# array containing 4D volume filenames and 
# baselines of respective 4D volumes
subjectNames=np.array([
   #['image_name001',5],
   #['image_name002',8],
   #['image_name007',10],
    ]);
                   
# extract filenames (pNameList) and (baseLineList)
pNameList = subjectNames[:,0];
baseLineList = subjectNames[:,1];

# path to excel sheet containing temporal information 
# about 4D volumes in subjectNames
fileAddress='path-to/subjectDicomInfo_gh.xlsx';
subjectInfo=pd.read_excel(fileAddress, sheet_name = 'subjects',engine='openpyxl');
reconMethod='SCAN';

dx=64;dy=64;dz=32;
 
# ttSNAP = 23;
# ssSNAP = 16;

for s in range(len(pNameList)):
# for s in range(1):
     patientName = pNameList[s];
     print('Processing ' + patientName);
 
     baselineTime = baseLineList[s];
     kidneyNone, croppedData4DL, croppedData4DR, croppedData4DL_pcs, croppedData4DR_pcs, KML, KMR, copyVol4D0, rkmOri, lkmOri, BoxCopy, copyKM = processSingleSubjects_ha(patientName,subjectInfo,int(baselineTime));
     
     # check if left or right kidney is missing
     noLeft = 0;
     noRight = 0;
     
     if croppedData4DL == []:
         noLeft = 1;
     elif croppedData4DR == []:
         noRight = 1;

     if 1:
       
        if croppedData4DL == []:
            croppedData4DL2 = np.copy(croppedData4DR);
            croppedData4DL = np.copy(croppedData4DR);
            croppedData4DR2 = np.copy(croppedData4DR);
            KML = np.copy(KMR);
            print('No left kidney')
            
        elif croppedData4DR == []:
            croppedData4DR2 = np.copy(croppedData4DL); 
            croppedData4DR = np.copy(croppedData4DL); 
            croppedData4DL2 = np.copy(croppedData4DL);
            KMR = np.copy(KML);
            print('No right kidney')
            
        else:
            print('Left and right kidneys exists')
            
        # if kidneyNone!=1 and kidneyNone!=0:  
        # croppedData4DL2 = np.copy(croppedData4DL);   
        # croppedData4DR2 = np.copy(croppedData4DR);
        
        gatherRabel = np.zeros(croppedData4DR.shape);
        gatherLabel = np.zeros(croppedData4DL.shape);
        
        # keepBin = np.zeros(croppedData4DR.shape);
        # keepBinL = np.zeros(croppedData4DL.shape);
        keepBinL2 = np.zeros(croppedData4DL.shape);
        
        closedKMR = np.zeros(KMR.shape);
        closedKML = np.zeros(KML.shape);
                
        closed_labels = np.zeros(croppedData4DL.shape);
        closed_rabels = np.zeros(croppedData4DR.shape);
        
        getRangei = croppedData4DL.shape[2];
        first30P = int(0.30*getRangei);
        last70P = int(0.70*getRangei);
        
        for tt in range(croppedData4DL.shape[3]):
        # for tt in range(7,13):    
            for ss in range(croppedData4DL.shape[2]):
            # for ss in range(15,18):
                
                # working on the left
                imageL = croppedData4DL[:,:,ss,tt]
                
                whilei = 1;
                while whilei < int(croppedData4DL.shape[3]):
                    if (imageL.max() == 0 and  imageL.min() == 0):
                        print('came into left kidney')
                        imageL = croppedData4DL[:,:,ss,(whilei+1)]
                    else:
                        break
                        
                    whilei += 1;

                
                # left kidney thresholding test
                imageL1 = np.copy(imageL); 
                # sort the flattened array
                intensitiesL = np.sort(imageL1, axis=None); 
                intensitiesL = intensitiesL[intensitiesL !=0];
                intensitiesLU = np.unique(intensitiesL);
                totLenL = len(intensitiesLU);
                totLenL25 = int(0.25*totLenL);
                if totLenL25 == 0:
                     totLenL25 = 2;
                intensitiesLCopy = np.copy(intensitiesLU)
                intensitiesLMins = intensitiesLCopy[0:totLenL25];
                minL = np.min(intensitiesLMins);
                maxL = np.max(intensitiesLMins);
 

                imageL2C = np.copy(imageL1);
                for ii in range(64):
                    for jj in range(64):
                         if imageL1[ii,jj] >= minL and imageL1[ii,jj] <= maxL and ss >= first30P and ss <= last70P: # fine-tune
                             imageL1[ii,jj] = imageL2C[ii,jj] - (minL);
                             
                         elif imageL1[ii,jj] >= minL and imageL1[ii,jj] <= maxL and (ss < first30P or ss > last70P): # fine-tune
                             imageL1[ii,jj] = imageL2C[ii,jj] - (minL);
            
                
#                 if tt == ttSNAP and ss == ssSNAP:
             
# #                    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
# #                    axes = ax0, ax1
# #                
# #                    ax0.imshow(imageL)
# #                    
# #                    ax0.set_title('Original slice')
# #                    ax1.imshow(imageL1)
# #                    ax1.set_title('Minimum subtracted slice')
# #                
# #                    for ax in axes:
# #                        ax.axis('off')
# #                           
# #                    plt.gray()
# #                    plt.show()
# #                    plt.draw() 
                    
#                     plt.figure(figsize=(3, 3))
#                     plt.imshow(imageL)
#                     plt.gray()
#                     plt.title('Original slice')
#                     #plt.show()
#                     #plt.draw()
                    
#                     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) + '_O'
#                     #plt.savefig(pathToSave + '.png')
#                     plt.savefig(pathToSave + '.pdf')
                    
#                     plt.figure(figsize=(3, 3))
#                     plt.imshow(imageL1)
#                     plt.gray()
#                     plt.title('Minimum subtracted slice')
#                     #plt.show()
#                     #plt.draw()
                    
#                     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_S'
#                     #plt.savefig(pathToSave + '.png')
#                     plt.savefig(pathToSave + '.pdf')
                    

                if np.isnan(imageL1).any() ==  1:
                    imageL1Temp = np.copy(imageL);
                    imageL1 = np.copy(imageL1Temp);
                    print("checking: left NaN")
                    
                if ss < first30P or ss > last70P:
                   xs1,ys1 = np.where(imageL1 != 0) 
                   imageL20 = np.copy(imageL1)
                   imageL2 = imageL20[min(xs1):max(xs1)+1,min(ys1):max(ys1)+1]
                   imageL2 = cv2.pow(imageL2,1.5) 
                   imageL1[min(xs1):max(xs1)+1,min(ys1):max(ys1)+1]= imageL2
                   
                imageL3 = imageL1-np.min(np.min(imageL1));
                imageL30 = imageL3/np.max(np.max(imageL3));
                imageL31 = imageL30.copy();

                if np.isnan(imageL31).any() ==  1:
                    imageL31 = np.copy(imageL1);
                    print("checking: left NaN")

                gain = 2; cutOff = 1.5;
                xLeft = gain*(cutOff-imageL31)
                left_imagex = 1/(1 + np.exp(xLeft)); 
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(imageL31)
                #     plt.gray()
                #     plt.title('imageL30 - cv2POW')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_L_cv2POW'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(left_imagex)
                #     plt.gray()
                #     plt.title('left_imagex - CE')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_L_CE'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                
                global_thresh = threshold_otsu(left_imagex)
                binary_global = left_imagex > global_thresh
                binary_global = binary_global.astype(int)
                label_im = np.copy(binary_global)
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(label_im)
                #     plt.gray()
                #     plt.title('label_im')
                #     #plt.show()
                #     #plt.draw()
            
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) + '_label_im'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                
                if kidneyNone == 1 or kidneyNone == 0:
                    
                    # right kidney thresholding test
                    imageR = croppedData4DR2[:,:,ss,tt]; 
                    
                    whilei = 1;
                    while whilei < int(croppedData4DR2.shape[3]):
                        if (imageR.max() == 0 and  imageR.min() == 0):
                            imageR = croppedData4DR2[:,:,ss,(whilei+1)]
                        else:
                            break
                            
                        whilei += 1;
                        
                else:
                        # right kidney thresholding test
                        imageR = croppedData4DR[:,:,ss,tt]; 
                        
                        whilei = 1;
                        while whilei < int(croppedData4DR.shape[3]):
                            if (imageR.max() == 0 and  imageR.min() == 0):
                                imageR = croppedData4DR[:,:,ss,(whilei+1)]
                            else:
                                break
                                
                            whilei += 1;
                         
#                if(imageR.max() == 0 and  imageR.min() == 0):
#                    print("came right");
#                    #imageR[1:32:,1:32] = imageR[1:32:,1:32] + 0.5;
#                    imageR = croppedData4DR2[:,:,ss,(tt-1)]
                
                #imageR1 = cv2.pow(imageR,1.5)
                imageR1 = np.copy(imageR); 
                #imageR = (255.0 / imageR.max() * (imageR - imageR.min())).astype(np.uint8)
#               imageR = imageR-np.min(np.min(imageR));
#               imageR = imageR/np.max(np.max(imageR));
                
#                if np.isnan(imageR1).any() ==  1:
#                    imageR1 = imageR;
#                    print("nan");

                # imageR1 = np.copy(imageR); 
                # intensitiesR = imageR2.flatten();

                intensitiesR = np.sort(imageR1, axis=None); 
                #intensitiesR = sorted_array[::-1];
                intensitiesR = intensitiesR[intensitiesR !=0];
                intensitiesRU = np.unique(intensitiesR);
                totLenR = len(intensitiesRU);
                totLenR25 = int(0.25*totLenR);
                if totLenR25 == 0:
                     totLenR25 = 2;
                intensitiesRCopy = np.copy(intensitiesRU)
                intensitiesRMins = intensitiesRCopy[0:totLenR25];
                minR = np.min(intensitiesRMins);
                maxR = np.max(intensitiesRMins);
                 

                imageR2C = np.copy(imageR1);
                for ii in range(64):
                    for jj in range(64):
                        if imageR1[ii,jj] >= minR and imageR1[ii,jj] <= maxR and ss >= first30P and ss <= last70P: # fine-tune
                            imageR1[ii,jj] = imageR2C[ii,jj] - (minR);
                        elif imageR1[ii,jj] >= minR and imageR1[ii,jj] <= maxR and (ss < first30P or ss > last70P): # fine-tune
                            imageR1[ii,jj] = imageR2C[ii,jj] - (minR);
                            
#                if ss < first30P or ss > last70P:
#                    imageR1 = cv2.pow(imageR1,1.5)
                            
                # if tt==ttSNAP and ss==ssSNAP:
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(imageR)
                #     plt.gray()
                #     plt.title('imageR - original')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) + '_OR'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(imageR1)
                #     plt.gray()
                #     plt.title('imageR1 - subtracted')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) + '_SR'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                if ss < first30P or ss > last70P: 
                   xr1,yr1 = np.where(imageR1 !=0) 
                   imageR11 = np.copy(imageR1)
                   imageR0 = imageR11[min(xr1):max(xr1)+1,min(yr1):max(yr1)+1]
                   imageR2 = cv2.pow(imageR0,1.5)
                   imageR1[min(xr1):max(xr1)+1,min(yr1):max(yr1)+1]= imageR2
                   
                #imageR1 = np.copy(imageR);            
                right_image40 = imageR1-np.min(np.min(imageR1));
                right_image41 = right_image40/np.max(np.max(right_image40));
                right_image42 = right_image41.copy();
                
                if np.isnan(right_image42).any() ==  1:
                    right_image42 = np.copy(imageR1);
                                         
                gain = 3; cutOff = 1.5;                
                yRight = gain*(cutOff-right_image42)
                imageR2 = 1/(1 + np.exp(yRight));
                
                # if tt==ttSNAP and ss==ssSNAP:
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(right_image41)
                #     plt.gray()
                #     plt.title('right_image41 - cv2POW')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_Rcv2POW'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(imageR2)
                #     plt.gray()
                #     plt.title('imageR2 - CE')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_RCE'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
   
                global_threshR = threshold_otsu(imageR2);
                binary_globalR = imageR2 > global_threshR;
                binary_globalR = binary_globalR.astype(int);
                rabel_im = np.copy(binary_globalR);
                
                # if tt==ttSNAP and ss==ssSNAP:
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(rabel_im)
                #     plt.gray()
                #     plt.title('rabel_im')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_rabel_im'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                # closing the labels
                closed_label_im = ndimage.binary_fill_holes(label_im).astype(int);
                closed_rabel_im = ndimage.binary_fill_holes(rabel_im).astype(int)
                closed_labels[:,:,ss,tt] = closed_label_im;
                closed_rabels[:,:,ss,tt] = closed_rabel_im;
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(closed_label_im)
                #     plt.gray()
                #     plt.title('closed_label_im')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_closed_label_im'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(closed_rabel_im)
                #     plt.gray()
                #     plt.title('closed_rabel_im')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_closed_rabel_im'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                maskL = KML[:,:,ss]
                maskL = maskL.astype(int)
                maskR = KMR[:,:,ss];
                maskR = maskR.astype(int)
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(maskL)
                #     plt.gray()
                #     plt.title('maskL')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_maskL'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(maskR)
                #     plt.gray()
                #     plt.title('maskR')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_maskR'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                # closing the masks                
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
                # closedL = cv2.morphologyEx(maskL,cv2.MORPH_OPEN,kernel)
                closedL = ndimage.binary_fill_holes(maskL).astype(int)
                closedKML[:,:,ss] = closedL;
                
                #closedR = cv2.morphologyEx(maskR,cv2.MORPH_OPEN,kernel)
                closedR = ndimage.binary_fill_holes(maskR).astype(int)
                closedKMR[:,:,ss] = closedR;
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(closedL)
                #     plt.gray()
                #     plt.title('closedL - mask')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_closed MaskL'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(closedR)
                #     plt.gray()
                #     plt.title('closedR - mask')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_closed MaskR'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                
                # experimenting with k-means clustering
                # imageR2[maskR==0]=0.0 
                # imageR2 = (255.0 / imageR2.max() * (imageR2 - imageR2.min())).astype(np.uint8)
                
                #imageR2 = to_rgb1(imageR2)
                # Z = imageR2.reshape((-1,2))
                # Z = np.float32(Z)

                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                # K = 6
                # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                
                # center = np.uint8(center)
                # res = center[label.flatten()]
                # res2 = res.reshape((imageR2.shape))
                # res2 = res2.astype(int)
                
                # keepBin[:,:,ss,tt] = res2;
                
#               closed_label_im = ndimage.binary_fill_holes(label_im).astype(int);
#               closed_rabel_im = ndimage.binary_fill_holes(rabel_im).astype(int)
#               closed_labels[:,:,ss] = closed_label_im;
#               closed_rabels[:,:,ss] = closed_rabel_im;
                
                # experimenting with k-means clustering for the left kidney
                # left_imagex = (255.0 / left_imagex.max() * (left_imagex - left_imagex.min())).astype(np.uint8)
                # left_imagex2 = np.copy(left_imagex);
                
                # Z1 = left_imagex.reshape((-1,2))
                # Z1 = np.float32(Z1)

                # criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                # K1 = 6
                # ret1,label1,center1 = cv2.kmeans(Z1,K1,None,criteria1,10,cv2.KMEANS_RANDOM_CENTERS)
                
                # # re-convert into uint8
                # center1 = np.uint8(center1)
                # res1 = center1[label1.flatten()]
                # res11 = res1.reshape((left_imagex.shape))
                # res11 = res11.astype(int)
                
                # res11[maskL == 0]= 0.0 
                # keepBinL[:,:,ss,tt] = res11;
                
                # left_imagex2[maskL==0]=0.0 
                # Z2 = left_imagex2.reshape((-1,2))
                # Z2 = np.float32(Z2)

                # criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                # K2 = 6
                # ret2,label2,center2=cv2.kmeans(Z2,K2,None,criteria1,10,cv2.KMEANS_RANDOM_CENTERS)
                
                # re-convert into uint8
                # center2 = np.uint8(center2)
                # res22 = center2[label2.flatten()]
                # res22 = res22.reshape((left_imagex2.shape))
                # res22x = res22.astype(int)
                # res22x = res22x-np.min(np.min(res22x));
                # res22x = res22x/np.max(np.max(res22x));
                
                label_im[maskL==0]= 0 # mask out background
                rabel_im[maskR==0]= 0 # mask out background
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(label_im)
                #     plt.gray()
                #     plt.title('label_im - masked back')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_label_im_masked'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(rabel_im)
                #     plt.gray()
                #     plt.title('rabel_im - masked back')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_rabel_im_masked'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                for ii in range(64):
                    for jj in range(64):
                        if maskL[ii,jj] == 1 and label_im[ii,jj] == 0:
                            label_im[ii,jj] = 2; # medulla
                            
                for ii in range(64):
                    for jj in range(64):
                        if maskR[ii,jj] == 1 and rabel_im[ii,jj] == 0:
                            rabel_im[ii,jj] = 2; # medulla

                            
                label_im2 = np.copy(label_im);
                rabel_im2 = np.copy(rabel_im);
                
                label_im2[closedL==0]= 3 # background changed
                rabel_im2[closedR==0]= 3 # background changed
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(label_im2)
                #     plt.gray()
                #     plt.title('label_im2 - labels')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_label_im2'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(rabel_im2)
                #     plt.gray()
                #     plt.title('rabel_im2 - labels')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_rabel_im2'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                label_im = np.copy(label_im2);
                rabel_im = np.copy(rabel_im2);
               
#                closed_label_im2 = np.copy(closed_label_im)
#                closed_label_im2[closed_label_im2<=0]=0
#                closed_label_im2[closed_label_im2>=1]=1
#                
#                closedL2 = np.copy(closedL)
#                closedL2[closedL2<=0]=0
#                closedL2[closedL2>=1]=1
                
                
#                uniqueX, countsX = np.unique(label_im, return_counts=True)
#                ccX = len(countsX);
#                
#                if ccX == 4:
#                    for ii in range(64):
#                        for jj in range(64):
#                            if closedL[ii,jj] == 1 and closed_label_im[ii,jj] == 0:
#                                label_im[ii,jj] = 1;
#                
#                uniqueXR, countsXR = np.unique(rabel_im, return_counts=True)
#                ccXR = len(countsXR);
#                if ccXR == 4:
#                    for ii in range(64):
#                        for jj in range(64):
#                            if closedR[ii,jj] == 1 and closed_rabel_im[ii,jj] == 0:
#                                rabel_im[ii,jj] = 1;
                
                
                # keepBinL2[:,:,ss,tt] = res22x;
                
#                kernel = np.ones((3,3),np.uint8);
#                closedRM = closedKMR[:,:,ss];
#                gClosedRM2 = cv2.morphologyEx(closedRM, cv2.MORPH_GRADIENT, kernel)
#                rabel_im[gClosedRM2==1]=1;
#                
#                closedLM = closedKML[:,:,ss];
#                gClosedLM2 = cv2.morphologyEx(closedLM, cv2.MORPH_GRADIENT, kernel)
#                label_im[gClosedLM2==1]=1;
                
                
                label_im = ~label_im;
                rabel_im = ~rabel_im;
                
                # if tt==ttSNAP and ss==ssSNAP:
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(label_im)
                #     plt.gray()
                #     plt.title('label_im - labels inversed')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_label_iv'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')
                    
                #     plt.figure(figsize=(3, 3))
                #     plt.imshow(rabel_im)
                #     plt.gray()
                #     plt.title('rabel_im - labels inversed')
                #     #plt.show()
                #     #plt.draw()
                    
                #     pathToSave = '/home/usr/Documents/stepsMedCortex' + '/' + patientName + '_' + str(tt) + '_' + str(ss) +'_rabel_iv'
                #     #plt.savefig(pathToSave + '.png')
                #     plt.savefig(pathToSave + '.pdf')

                gatherLabel[:,:,ss,tt] = label_im
                gatherRabel[:,:,ss,tt] = rabel_im
                
        gatherLabel[gatherLabel == -4]= 0
        gatherLabel[gatherLabel == -3]= 1
        gatherLabel[gatherLabel == -2]= 2
        gatherLabel[gatherLabel == -1]= 3
        
        gatherRabel[gatherRabel == -4]= 0
        gatherRabel[gatherRabel == -3]= 1
        gatherRabel[gatherRabel == -2]= 2
        gatherRabel[gatherRabel == -1]= 3
        
        gatherRabel2 = np.copy(gatherRabel)   
        gatherLabel2 = np.copy(gatherLabel) 
        
        testLeft = np.zeros(KML.shape)
        testRight = np.zeros(KMR.shape)
        
        """
        #### experiment: new find ratio area and number of medulla over time
        for slaX in range(1):
        #for sla in range(gatherLabel.shape[2]):
            sla = 16
            
            numCels = []; numCelsR = [];
            ratios = []; ratiosR = [];
             
            for stt in range(gatherLabel.shape[3]):
                
                lSlice = np.copy(gatherLabel2[:,:,int(sla),int(stt)]);
                lSlice2 = np.copy(gatherLabel2[:,:,int(sla),int(stt)]);
                lSlice3 = np.copy(gatherLabel2[:,:,int(sla),int(stt)]);
                
                rSlice = np.copy(gatherRabel2[:,:,int(sla),int(stt)]);
                rSlice2 = np.copy(gatherRabel2[:,:,int(sla),int(stt)]);
                rSlice3 = np.copy(gatherRabel2[:,:,int(sla),int(stt)]);
                
                  
                # starting the left
                totArea = lSlice[lSlice != 0];
                totArea2 = lSlice2[lSlice2 != 0];
                    
                C_area = totArea[totArea == 2];
                M_area = totArea2[totArea2 == 1]; 
                
                p_CM = 100.0;
                if C_area.size != 0 and M_area.size != 0:
                    p_CM = (M_area.size/totArea.size)*100
                    
                # starting the right 
                totAreaR = rSlice[rSlice != 0];
                totArea2R = rSlice2[rSlice2 != 0];
                    
                CR_area = totAreaR[totAreaR == 2];
                MR_area = totArea2R[totArea2R == 1];         
                    
                p_CMR = 100.0;
                if CR_area.size != 0 and MR_area.size != 0:
                    p_CMR = (MR_area.size/totAreaR.size)*100
            
                ratios.append([p_CM,stt])  
                ratiosR.append([p_CMR,stt]) 
                
                lSlice3[lSlice3 >= 2] = 0; lSlice3[lSlice3 < 1] = 0;
                rSlice3[rSlice3 >= 2] = 0; rSlice3[rSlice3 < 1] = 0;
                 
                if np.sum(lSlice3) > 0: 
                     lSlice3 = ndimage.binary_fill_holes(lSlice3).astype(int)
                     lSlice3=morphology.remove_small_objects(lSlice3.astype(bool), min_size=5,in_place=True).astype(int);
                     
                     val = filters.threshold_otsu(lSlice3)
                     drops = (lSlice3 > val).astype('int')
#                    numDdrops = measure.label(drops)
                     labels, numDdrops = ndi.label(drops)                     
                else:
                     numDdrops = 0;
                     
                if np.sum(rSlice3) > 0:
                     rSlice3 = ndimage.binary_fill_holes(rSlice3).astype(int)
                     rSlice3=morphology.remove_small_objects(rSlice3.astype(bool), min_size=5,in_place=True).astype(int);
                     valR = filters.threshold_otsu(rSlice3)
                     dropsR = (rSlice3 > valR).astype('int')
#                    numDdropsR = measure.label(dropsR)
                     labelsR, numDdropsR = ndi.label(dropsR)                     
                else:
                     numDdropsR = 0;
                     
                numCels.append([numDdrops,stt])  
                numCelsR.append([numDdropsR,stt]) 
                    
                fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
                axes = ax0, ax1
            
                ax0.imshow(lSlice3)
                ax0.set_title('left ' + str(sla) + ' ' + str(stt))
                ax1.imshow(rSlice3)
                ax1.set_title('right ' + str(sla) + ' ' + str(stt))
                
                for ax in axes:
                   ax.axis('off')
                           
                plt.gray()
                plt.show()
                plt.draw()    
        
        
        """
        
        getRange = gatherLabel.shape[2];
        first25P = int(0.30*getRange);
        last25P = int(0.70*getRange)
        #for slicing in range(gatherLabel.shape[2]):
        for slicing in range(gatherLabel.shape[2]):
        #for slicing in range(1):
            
            #sii = 15;
            test = gatherLabel2[:,:,slicing,int(gatherLabel.shape[3]/2)]; # /4 # fine-tune
            testR = gatherRabel2[:,:,slicing,int(gatherRabel.shape[3]/2)]; # /4 # fine-tune
            
            for streaming in range(gatherLabel.shape[3]):
                
                lSlice = np.copy(gatherLabel2[:,:,int(slicing),int(streaming)]);
                lSlice2 = np.copy(gatherLabel2[:,:,int(slicing),int(streaming)]);

                
                rSlice = np.copy(gatherRabel2[:,:,int(slicing),int(streaming)]);
                rSlice2 = np.copy(gatherRabel2[:,:,int(slicing),int(streaming)]);
                
                uTest, cTest = np.unique(lSlice, return_counts=True)
                ccT = len(cTest);
                #print(str(ccT));
                
                uTestR, cTestR = np.unique(rSlice, return_counts=True)
                ccTR = len(cTestR);
                
                # starting the left kidney
                totArea = lSlice[lSlice != 0];
                totArea2 = lSlice2[lSlice2 != 0];
                    
                C_area = totArea[totArea == 2];
                M_area = totArea2[totArea2 == 1];         
                    
                subtractedL =  closedKML[:,:,slicing];
                
                if C_area.size != 0 and M_area.size != 0:

                    p_CM = (M_area.size/totArea.size)*100
                    
                    # fine-tune 
                    if ccT >= 3 and p_CM > 10.0 and p_CM <50.0 and slicing >= first25P and slicing <= last25P:
                        
                        for ii in range(64):
                            for jj in range(64):
                                if lSlice[ii,jj] == 1 and test[ii,jj] == 2: 
                                    test[ii,jj] = 1; #medulla
                    # fine-tune                
                    elif ccT >= 3 and p_CM < 50.0 and (slicing < first25P or slicing > last25P):
                        
                         for ii in range(64):
                            for jj in range(64):
                                if lSlice[ii,jj] == 1 and test[ii,jj] == 2: 
                                    test[ii,jj] = 1; #medulla
                    
#                    if streaming==ttSNAP and slicing==ssSNAP:
#                    
#                        plt.figure(figsize=(3, 3))
#                        plt.imshow(test)
#                        plt.gray()
#                        plt.title('test - medulla changed')
#                        #plt.show()
#                        #plt.draw()
#                        
#                        pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_test1'
#                        #plt.savefig(pathToSave + '.png')
#                        plt.savefig(pathToSave + '.pdf')
                        
                    
                    closedLM = closedKML[:,:,slicing];
                    closedLM2 = np.copy(closedLM);
                    closedLT = closed_labels[:,:,slicing,int(closed_labels.shape[3]/2)] # fine-tune
                    subtracted = closedLM-closedLT;
                    
                    kernel2 = np.ones((3,3),np.uint8);
                    gClosedLM2 = cv2.morphologyEx(closedLM2, cv2.MORPH_GRADIENT, kernel2)
                    subtractedL = np.copy(gClosedLM2);
                    
                    # fine-tune
                    if ccT <= 3: #and (slicing < first25P or slicing > last25P):
                        for ii in range(64):
                            for jj in range(64):
                                if closedLM[ii,jj] == 1 and closedLT[ii,jj] == 0 and test[ii,jj] == 1:
                                    test[ii,jj] = 2;
                                    
                    test[gClosedLM2==1]=2;
                      
                    # fine-tune
                    if ccT == 4:# and (slicing < first25P or slicing > last25P):
                        for ii in range(64):
                            for jj in range(64):
                                if closedLM[ii,jj] == 1 and closedLT[ii,jj] == 0 and test[ii,jj] == 1:
                                    test[ii,jj] = 2;
                                    
                    test[gClosedLM2==1]=2;
                    
#                    if streaming==ttSNAP and slicing==ssSNAP:
#                    
#                        plt.figure(figsize=(3, 3))
#                        plt.imshow(test)
#                        plt.gray()
#                        plt.title('test - medulla gClosedLM2')
#                        #plt.show()
#                        #plt.draw()
#                        
#                        pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_test2'
#                        #plt.savefig(pathToSave + '.png')
#                        plt.savefig(pathToSave + '.pdf')
                                    
                # starting the right kidney
                totAreaR = rSlice[rSlice != 0];
                totArea2R = rSlice2[rSlice2 != 0];
                    
                CR_area = totAreaR[totAreaR == 2];
                MR_area = totArea2R[totArea2R == 1];         
                    
                #subtractedR = closedKMR[:,:,slicing];
                if CR_area.size != 0 and MR_area.size != 0:

                    p_CMR = (MR_area.size/totAreaR.size)*100
                    
                    # fine-tune
                    if ccTR >= 3 and p_CMR > 5.0 and p_CMR < 50.0 and slicing >= first25P and slicing <= last25P:
                        
                        for ii in range(64):
                            for jj in range(64):
                                if rSlice[ii,jj] == 1 and testR[ii,jj] == 2: 
                                    testR[ii,jj] = 1; #medulla
                    # fine-tune                
                    elif ccTR >= 3 and p_CMR < 50.0 and (slicing < first25P or slicing > last25P):
                         for ii in range(64):
                            for jj in range(64):
                                if rSlice[ii,jj] == 1 and testR[ii,jj] == 2: 
                                    testR[ii,jj] = 1; #medulla
                    # fine-tune                
                    elif ccTR < 3:
                        print(str(ccTR) + "two unique labels" + str(p_CMR));
                        
                    
#                    if streaming==ttSNAP and slicing==ssSNAP:
#                    
#                        plt.figure(figsize=(3, 3))
#                        plt.imshow(testR)
#                        plt.gray()
#                        plt.title('testR - medulla')
#                        #plt.show()
#                        #plt.draw()
#                        
#                        pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) + '_testR1'
#                        #plt.savefig(pathToSave + '.png')
#                        plt.savefig(pathToSave + '.pdf')
            
                    closedRM = closedKMR[:,:,slicing];
                    closedRM2 = np.copy(closedRM);
                    closedRT = closed_rabels[:,:,slicing,int(closed_rabels.shape[3]/2)]
                    #closedRT = closed_rabels[:,:,slicing,0]
                    #subtractedR = closedRM-closedRT;
                    
                    kernel = np.ones((3,3),np.uint8);
                    gClosedRM2 = cv2.morphologyEx(closedRM2, cv2.MORPH_GRADIENT, kernel)
                    
                    if ccTR == 3:# and (slicing < first25P or slicing > last25P):# and p_CMR < 90.0:
                        for ii in range(64):
                            for jj in range(64):
                                #if closedRM[ii,jj] == 1 and closedRT[ii,jj] == 0 and testR[ii,jj] == 1:
                                if closedRM[ii,jj] == 1 and closedRT[ii,jj] == 0 and testR[ii,jj] == 1:
                                    testR[ii,jj] = 2;
                                    
                    testR[gClosedRM2==1]= 2;               
#                   elif ccTR == 3:
#                        testR[gClosedRM2==1]=2;
                                    
                    if ccTR == 4:# and (slicing < first25P or slicing > last25P): # and p_CMR < 90.0:
                        for ii in range(64):
                            for jj in range(64):
                                if closedRM[ii,jj] == 1 and closedRT[ii,jj] == 0 and testR[ii,jj] == 1:
                                    testR[ii,jj] = 2;
#                                    
                    testR[gClosedRM2 == 1] = 2;
#                    elif ccTR == 4:
#                        testR[gClosedRM2==1]=2;
                        
                        
                    #subtractedR = np.copy(gClosedRM2);
                    #subtractedL = np.copy(gClosedLM2);
                    
#                    if streaming==ttSNAP and slicing==ssSNAP:
#                    
#                        plt.figure(figsize=(3, 3))
#                        plt.imshow(testR)
#                        plt.gray()
#                        plt.title('testR - medulla gClosedRM2')
#                        #plt.show()
#                        #plt.draw()
#                        
#                        pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) + '_testR2'
#                        #plt.savefig(pathToSave + '.png')
#                        plt.savefig(pathToSave + '.pdf')
                                       
            # working on the left kidney
            testLMed = np.copy(test);
            testLMed[testLMed==2]= 0;
            testLMed[testLMed==3]= 0;
            
            if np.sum(testLMed) != 0:
                #testLMed = ndimage.binary_fill_holes(testLMed).astype(int)
                testLMed=morphology.remove_small_objects(testLMed.astype(bool), min_size=5,in_place=True).astype(int);
                
            # if slicing==ssSNAP:
                
            #     plt.figure(figsize=(3, 3))
            #     plt.imshow(testLMed)
            #     plt.gray()
            #     plt.title('testLMed')
            #     #plt.show()
            #     #plt.draw()
                
            #     pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_testLMed'
            #     #plt.savefig(pathToSave + '.png')
            #     plt.savefig(pathToSave + '.pdf')
                
            for ii in range(64):
                for jj in range(64):
                    if testLMed[ii,jj] == 0 and test[ii,jj] == 1:
                        test[ii,jj] = 2;
                        
            testLeft[:,:,slicing] = test;
            
            # if slicing==ssSNAP:
            
            #     plt.figure(figsize=(3, 3))
            #     plt.imshow(test)
            #     plt.gray()
            #     plt.title('test after testLMed')
            #     #plt.show()
            #     #plt.draw()
                
            #     pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_test3'
            #     #plt.savefig(pathToSave + '.png')
            #     plt.savefig(pathToSave + '.pdf')


            # working on the right kidney
            testRMed = np.copy(testR);
            testRMed[testRMed==2]= 0;
            testRMed[testRMed==3]= 0;
            
            # if slicing==ssSNAP:
            
            #     plt.figure(figsize=(3, 3))
            #     plt.imshow(testRMed)
            #     plt.gray()
            #     plt.title('testRMed')
            #     #plt.show()
            #     #plt.draw()
                
            #     pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_testRMed'
            #     #plt.savefig(pathToSave + '.png')
            #     plt.savefig(pathToSave + '.pdf')
            
            if np.sum(testRMed) != 0:
                testRMed=morphology.remove_small_objects(testRMed.astype(bool), min_size=5,in_place=True).astype(int);
                        
            for ii in range(64):
                for jj in range(64):
                    if testRMed[ii,jj] == 0 and testR[ii,jj] == 1:
                        testR[ii,jj] = 2;
            
            testRight[:,:,slicing] = testR;
            
            # if slicing==ssSNAP:
            
            #     plt.figure(figsize=(3, 3))
            #     plt.imshow(testR)
            #     plt.gray()
            #     plt.title('testR after testRMed')
            #     #plt.show()
            #     #plt.draw()
                
            #     pathToSave = '/home/usr/Documents/stepsMedCortex2' + '/' + patientName + '_' + str(streaming) + '_' + str(slicing) +'_testR3'
            #     #plt.savefig(pathToSave + '.png')
            #     plt.savefig(pathToSave + '.pdf')
                
            ## plot segmentation to check results ##
            #uniqueTest, countsTest = np.unique(testLeft[:,:,slicing], return_counts=True)
            #ccTest = len(countsTest);

#            uniqueTestL, countsTestL = np.unique(testLeft[:,:,slicing], return_counts=True)
#            ccTestL = len(countsTestL);
#         
#            fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
#            axes = ax0, ax1
#            
#            #plt.figure(figsize=(3, 3))
#            ax0.imshow(testLeft[:,:,slicing])
#            #plt.gray()
#            ax0.set_title('left ' + str(slicing) + ' ' + str(ccTestL))
#            #plt.show()
#            #plt.draw()
#             
#            #plt.figure(figsize=(3, 3))
#            ax1.imshow(testRight[:,:,slicing])
#            #plt.gray()
#            ax1.set_title('right ')
#            
#            for ax in axes:
#               ax.axis('off')
#                       
#            plt.gray()
#            plt.show()
#            plt.draw()        
     

           
     #for si in range(croppedData4DL.shape[2]):
     for sii in range(1):
         for ti in range(croppedData4DR.shape[3]):
             #for ti in range(35,45):
             si = 16;
             #fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(4, 4))
             fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
             axes = ax0, ax1, ax2, ax3
             
     #               plt.figure(figsize=(2, 2))
     #               plt.imshow(gatherLabel[:,:,si,ti])
     #               plt.gray()
     #               plt.title('slice '+ str(si)+ ' time' + str(ti))
     #               plt.show()
     #               plt.draw()
             
     #           fig, axes = plt.subplots(nrows=3, figsize=(6, 7))
     #           ax0, ax1, ax2= axes
     #           plt.gray()
     #            
             #ax0.imshow(binary_global)
             #showing = gatherLabel[:,:,si,ti];
             
             uniqueX, countsX = np.unique(gatherLabel2[:,:,si,ti], return_counts=True)
             ccX = len(countsX);
             
             uniqueX2, countsX2 = np.unique(keepBinL2[:,:,si,ti], return_counts=True)
             ccX2 = len(countsX2);
             
             uniqueXR, countsXR = np.unique(gatherRabel2[:,:,si,ti], return_counts=True)
             ccXR = len(countsXR);
             
             ax0.imshow(gatherLabel2[:,:,si,ti])
             #ax0.imshow(keepBinL[:,:,si,ti])
             ax0.set_title(str(ccX) + ' left ' + str(si) + ' ' + str(ti))
              
             #ax1.imshow(keepBin[:,:,si,ti])
             #ax1.set_title(str(ccXR) + ' right ' + str(si) + ' ' + str(ti))
             
             #ax0.imshow(gatherLabel[:,:,si,ti])
             #ax2.imshow(keepBinL2[:,:,si,ti])
             #ax2.set_title(str(ccX2) + ' left ' + str(si) + ' ' + str(ti))
             
             # ax1.imshow((gatherRabel[:,:,si,ti])
             #ax1.set_title('right ' + str(si) + ' ' + str(ti))
             
             ax1.imshow(croppedData4DL[:,:,si,ti])
             #ax2.imshow(KML[:,:,si])
             #ax2.imshow(closed_labels[:,:,si])
             ax1.set_title('left ' + str(si) + ' ' + str(ti))
             
             ax2.imshow(gatherRabel2[:,:,si,ti])
             #ax3.imshow(KMR[:,:,si])
             #ax3.imshow(closed_rabels[:,:,si])
             ax3.set_title(str(ccXR) + ' right ' + str(si) + ' ' + str(ti))
             
             ax3.imshow(croppedData4DR[:,:,si,ti])
             #ax3.imshow(KMR[:,:,si])
             #ax3.imshow(closed_rabels[:,:,si])
             ax3.set_title('right ' + str(si) + ' ' + str(ti))
              
             for ax in axes:
                 ax.axis('off')
    
             plt.gray()
             plt.show()
             plt.draw()
         
 
         # re-insert into original position
         xyDim = 224; #224
         zDim=copyVol4D0.shape[2]
         predMaskR=np.zeros((1,xyDim,xyDim,zDim));
         predMaskL=np.zeros((1,xyDim,xyDim,zDim));
    
         _, originalMask, _, _, _ = funcs_mc.readData4(patientName,subjectInfo,reconMethod,1);
    
         s = 0;
        
         testLeftC = np.copy(testLeft);
         testLeftM = np.copy(testLeft);
         testLeftP = np.copy(testLeft);
        
         testRightC = np.copy(testRight);
         testRightM = np.copy(testRight);
         testRightP = np.copy(testRight);
        
         testLeftM[testLeftM==2]=0;
         testLeftM[testLeftM==3]=0;
         testRightM[testRightM==2]=0;
         testRightM[testRightM==3]=0;
               
         # to contain resulting segmentation for both left and right
         labels_pred_2 = np.zeros((2,64,64,32)); 
         leftKeeping = np.copy(testLeftM)
         leftKeeping2 = np.copy(testLeftM)
        
         # to contain resulting segmentation for both left and right
         labels_pred_2R = np.zeros((2,64,64,32)); 
         leftKeepingR = np.copy(testRightM)
         leftKeeping2R = np.copy(testRightM) 
        
         labels_pred_2[2*s,:,:,:] = leftKeeping[:,:,:];
         labels_pred_2[2*s+1,:,:,:] = leftKeeping2[:,:,:];
        
         labels_pred_2R[2*s,:,:,:] = leftKeepingR[:,:,:];
         labels_pred_2R[2*s+1,:,:,:] = leftKeeping2R[:,:,:];
    
         boxTrue = np.copy(BoxCopy);
         Box=np.reshape(boxTrue,[2,6]).astype('int');
    
         originalMask[originalMask>1]=1;
         if Box[0,2]+ Box[0,5]+ 3 >= originalMask.shape[2] or Box[0,2]+Box[0,5]-3 < 0:
            Box[:,[3,4,5]]= Box[:,[3,4,5]]+[10,10,0];
         else:
            Box[:,[3,4,5]]= Box[:,[3,4,5]]+[10,10,3];
        
         if kidneyNone!=0:
            # right kidney exists
            Rk=labels_pred_2R[2*s,:,:,:]
            croppedData4DR=signal.resample(Rk,Box[0,3], t=None, axis=0);
            croppedData4DR=signal.resample(croppedData4DR,Box[0,4], t=None, axis=1);
            croppedData4DR=signal.resample(croppedData4DR,Box[0,5], t=None, axis=2);
            croppedData4DR[croppedData4DR>=0.5]=2;croppedData4DR[croppedData4DR<0.5]=0
            #croppedData4DR[croppedData4DR==0]=1;croppedData4DR[croppedData4DR==2]=0  
            
            predMaskR[s,int(Box[0,0]-Box[0,3]/2):int(Box[0,0]+Box[0,3]/2),\
                                int(Box[0,1]-Box[0,4]/2):int(Box[0,1]+Box[0,4]/2),\
                                int(Box[0,2]-Box[0,5]/2):int(Box[0,2]+Box[0,5]/2)]=croppedData4DR;
                                    
         if kidneyNone!=1:     
            # left kidney exists
            Lk=labels_pred_2[2*s+1,:,:,:]
            croppedData4DL=signal.resample(Lk,Box[1,3], t=None, axis=0);
            croppedData4DL=signal.resample(croppedData4DL,Box[1,4], t=None, axis=1);
            croppedData4DL=signal.resample(croppedData4DL,Box[1,5], t=None, axis=2);
            croppedData4DL[croppedData4DL>=0.5]=2; croppedData4DL[croppedData4DL<0.5]=0
            #croppedData4DL[croppedData4DL>0.5]=2; croppedData4DL[croppedData4DL<0.5]=0
            #croppedData4DL[croppedData4DL==0]=1;croppedData4DL[croppedData4DL==2]=0    
            
            predMaskL[s,int(Box[1,0]-Box[1,3]/2):int(Box[1,0]+Box[1,3]/2),\
                                int(Box[1,1]-Box[1,4]/2)+1:int(Box[1,1]+Box[1,4]/2)+1,\
                                int(Box[1,2]-Box[1,5]/2):int(Box[1,2]+Box[1,5]/2)]=croppedData4DL;    
        
        
         if np.sum(predMaskL) != 0:
            predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
         if np.sum(predMaskR) != 0:
            predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);
            
         Masks2Save={};
         Masks2SaveCortex={};
       
         predMaskL2 = np.copy(predMaskL)
         predMaskL2[predMaskL2==2]=1; 
         predMaskL2 = ndimage.binary_fill_holes(predMaskL2).astype(int)
        
         predMaskR2 = np.copy(predMaskR)
         predMaskR2[predMaskR2==2]=1; 
         predMaskR2 = ndimage.binary_fill_holes(predMaskR2).astype(int)
            
         zDimOrig = zDim
         predMaskR3=zoom(predMaskR2[s,:,:,:],(1,1,zDimOrig/zDim),order=0);
         predMaskL3=zoom(predMaskL2[s,:,:,:],(1,1,zDimOrig/zDim),order=0);
         
         
         predMaskL3[originalMask==0]=0;
         lkmOri[lkmOri==2]=1
         lkmOri[predMaskL3==1]=0
        
         predMaskR3[originalMask==0]=0;
         rkmOri[rkmOri==2]=1
         rkmOri[predMaskR3==1]=0
      
         # if only one kidney is present
         if noLeft == 1:
            predMaskL3 = np.zeros(predMaskR3.shape);
            lkmOri =  np.zeros(rkmOri.shape);
         elif noRight == 1:
            predMaskR3 = np.zeros(predMaskL3.shape);
            rkmOri =  np.zeros(lkmOri.shape);

        
         Masks2Save['R']=np.copy(predMaskR3.astype(float));
         Masks2Save['L']=np.copy(predMaskL3.astype(float));
    
         Masks2SaveCortex['R']=np.copy(rkmOri.astype(float));
         Masks2SaveCortex['L']=np.copy(lkmOri.astype(float));
         
         print(patientName)
         pathToFolder='path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/';
         if not os.path.exists(pathToFolder):
             os.makedirs(pathToFolder)
        
         reconMethod='SCAN'; 
         overwrite = 1;
         funcs_mc.writeMasksMulti(patientName,subjectInfo,reconMethod,Masks2Save,overwrite,1);
         funcs_mc.writeMasksMulti(patientName,subjectInfo,reconMethod,Masks2SaveCortex,overwrite,0);
 
