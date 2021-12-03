#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import os, sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
# import SimpleITK as sitk
# from matplotlib.widgets import Slider, Button, RadioButtons
# import matplotlib.pyplot as plt

def readData4(patientName,subjectInfo,reconMethod,genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    

    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';
        
    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im=img.get_data()  
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data()  
            
            
            if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii.gz'):
                lkm1= nib.load(maskAddress+'leftKidneyMask_automatic.nii.gz');
                lkm=2*lkm1.get_data()
                lkm[lkm==4]=2
                lkm[lkm==6]=2
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                lkm=lkmUpper+lkmLower;
            else:
                lkm=np.zeros(np.shape(am));
                
            if os.path.isfile(maskAddress+'rightKidneyMask_automatic.nii.gz'):
                rkm1= nib.load(maskAddress+'rightKidneyMask_automatic.nii.gz');
                rkm=rkm1.get_data() 
                rkm[rkm==3]=1
                rkm[rkm==2]=1
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper_automatic.nii.gz'):
                rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper_automatic.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower_automatic.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                rkm=rkmUpper+rkmLower;
            else:
                rkm=np.zeros(np.shape(am));
                            
#            if np.sum(rkm[:])==0:
#                rkm=lkm;
#            if np.sum(lkm[:])==0:
#                lkm=rkm;
        else:
            lkm=0;rkm=0;am=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()  

        # x=subjectInfo['timeRes'][patientName];
        x0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']       
        index = x0.index.values
        index = index[0]
        x = x0.loc[index]

        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);
        if seq2tres>seq1tres:
            num2=seq2tres/seq1tres;
            im3=zoom(im2,(1,1,1,num2),order=0);
            im=np.concatenate((im1, im3), axis=3);
        else:
            im=np.concatenate((im1, im2), axis=3);
            
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            if os.path.isfile(maskAddress+'leftKidneyMask_auto.nii.gz'):
                    lkm1= nib.load(maskAddress+'leftKidneyMask_automatic.nii.gz'); lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                    lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                    lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                    lkm=lkmUpper+lkmLower;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddress+'rightKidneyMask_automatic.nii.gz'):
                    rkm1= nib.load(maskAddress+'rightKidneyMask_automatic.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                    rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=1*rkm1_upper.get_data() 
                    rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=1*rkm1_lower.get_data() 
                    rkm=rkmUpper+rkmLower;
            else:
                    rkm=np.zeros(np.shape(am));      
        else:
            lkm=0;rkm=0;am=0;
    
    
    boxes=[];
    if genBoundBox:
        
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);
        
        # bounding box center + width and height
        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    oriKM = rkm+lkm;
    im=(im/np.amax(im))*100;
    
    return im, oriKM, boxes, rkm,lkm


def readDataCMP_baseline(patientName,subjectInfo,reconMethod,genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]
    
    maskAddress2 = 'path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/'+patientName+'/';
    maskAddress3 = 'path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/'+patientName+'/';

    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes/'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes/'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im = img.get_data()  
        
        if os.path.isfile(maskAddress3+'leftKidneyMask_baseline_m.nii.gz'):
            lkmed1= nib.load(maskAddress3+'leftKidneyMask_baseline_m.nii.gz');
            lkmed=2*lkmed1.get_data()
            lkcor1= nib.load(maskAddress3+'leftKidneyMask_baseline_c.nii.gz');
            lkcor=2*lkcor1.get_data()
        else:
            lkmed = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            lkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress3+'rightKidneyMask_baseline_m.nii.gz'):
            rkmed1= nib.load(maskAddress3+'rightKidneyMask_baseline_m.nii.gz');
            rkmed=rkmed1.get_data()
            rkcor1= nib.load(maskAddress3+'rightKidneyMask_baseline_c.nii.gz');
            rkcor=rkcor1.get_data()
        else:
           rkmed = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
           rkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');
            am=am1.get_data()  
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');
                lkm=2*lkm1.get_data()
                lkm[lkm==4]=2
                lkm[lkm==6]=2
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                lkm=lkmUpper+lkmLower;
            else:
                lkm=np.zeros(np.shape(am));
                
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');
                rkm=rkm1.get_data()  
                rkm[rkm==3]=1
                rkm[rkm==2]=1
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                rkm=rkmUpper+rkmLower;
            else:
                rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;rkmed=0;lkmed=0;rkcor=0;lkcor=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()  

        # x=subjectInfo['timeRes'][patientName];
        x0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']       
        index = x0.index.values
        index = index[0]
        x = x0.loc[index]
        
        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);

        num2=seq2tres/seq1tres;
        im3=zoom(im2,(1,1,1,num2),order=0);
        im=np.concatenate((im1, im3), axis=3);
        
        
        if os.path.isfile(maskAddress2+'leftKidneyMask_baseline_m.nii.gz'):
            lkmed1= nib.load(maskAddress2+'leftKidneyMask_baseline_m.nii.gz');
            lkmed=2*lkmed1.get_data()
            lkcor1= nib.load(maskAddress2+'leftKidneyMask_baseline_c.nii.gz');
            lkcor=2*lkcor1.get_data()
        else:
            lkmed=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            lkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress2+'rightKidneyMask_baseline_m.nii.gz'):
            rkmed1= nib.load(maskAddress2+'rightKidneyMask_baseline_m.nii.gz');
            rkmed=rkmed1.get_data()
            rkcor1= nib.load(maskAddress2+'rightKidneyMask_baseline_c.nii.gz');
            rkcor=rkcor1.get_data()
        else:
            rkmed=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            rkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                    lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                    lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                    lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                    lkm=lkmUpper+lkmLower;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                    rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                    rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                    rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                    rkm=rkmUpper+rkmLower;
            else:
                    rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;

    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    oriKM = rkm+lkm;
    corOri = rkcor+lkcor;
    medOri = rkmed+lkmed;
    im=(im/np.amax(im))*100;
    
    return im, oriKM, boxes, rkm, lkm, lkmed, rkmed, medOri, lkcor, rkcor, corOri, seqNum


def readDataCMP(patientName,subjectInfo,reconMethod,genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]
    
    maskAddress2 = 'path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/';
    maskAddress3 = 'path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/';

    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im = img.get_data()  
        
        if os.path.isfile(maskAddress3+'leftKidneyMask_automatic_m.nii.gz'):
            lkmed1= nib.load(maskAddress3+'leftKidneyMask_automatic_m.nii.gz');
            lkmed=2*lkmed1.get_data()
            lkcor1= nib.load(maskAddress3+'leftKidneyMask_automatic_c.nii.gz');
            lkcor=2*lkcor1.get_data()
        else:
            lkmed = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            lkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress3+'rightKidneyMask_automatic_m.nii.gz'):
            rkmed1= nib.load(maskAddress3+'rightKidneyMask_automatic_m.nii.gz');
            rkmed=rkmed1.get_data()
            rkcor1= nib.load(maskAddress3+'rightKidneyMask_automatic_c.nii.gz');
            rkcor=rkcor1.get_data()
        else:
           rkmed = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
           rkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');
            am=am1.get_data()  
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');
                lkm=2*lkm1.get_data()
                lkm[lkm==4]=2
                lkm[lkm==6]=2
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                lkm=lkmUpper+lkmLower;
            else:
                lkm=np.zeros(np.shape(am));
                
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');
                rkm=rkm1.get_data()  
                rkm[rkm==3]=1
                rkm[rkm==2]=1
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                rkm=rkmUpper+rkmLower;
            else:
                rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;rkmed=0;lkmed=0;rkcor=0;lkcor=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()  

        # x=subjectInfo['timeRes'][patientName];
        x0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']       
        index = x0.index.values
        index = index[0]
        x = x0.loc[index]
        
        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);
        
        
        num2=seq2tres/seq1tres;
        im3=zoom(im2,(1,1,1,num2),order=0);
        im=np.concatenate((im1, im3), axis=3);

        if os.path.isfile(maskAddress2+'leftKidneyMask_automatic_m.nii.gz'):
            lkmed1= nib.load(maskAddress2+'leftKidneyMask_automatic_m.nii.gz');
            lkmed=2*lkmed1.get_data()
            lkcor1= nib.load(maskAddress2+'leftKidneyMask_automatic_c.nii.gz');
            lkcor=2*lkcor1.get_data()
        else:
            lkmed=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            lkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress2+'rightKidneyMask_automatic_m.nii.gz'):
            rkmed1= nib.load(maskAddress2+'rightKidneyMask_automatic_m.nii.gz');
            rkmed=rkmed1.get_data()
            rkcor1= nib.load(maskAddress2+'rightKidneyMask_automatic_c.nii.gz');
            rkcor=rkcor1.get_data()
        else:
            rkmed=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            rkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                    lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                    lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                    lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                    lkm=lkmUpper+lkmLower;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                    rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                    rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                    rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                    rkm=rkmUpper+rkmLower;
            else:
                    rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;

    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    oriKM = rkm+lkm;
    corOri = rkcor+lkcor;
    medOri = rkmed+lkmed;
    im=(im/np.amax(im))*100;
    
    return im, oriKM, boxes, rkm, lkm, lkmed, rkmed, medOri, lkcor, rkcor, corOri, seqNum

def readDataCMP_GT(patientName,subjectInfo,reconMethod,genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]
    
    maskAddress2 = 'path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
    maskAddress3 = 'path-to-folder-containing-4D-volumes/'+patientName+'/';

    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes/'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes/'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im = img.get_data()  
        
        if os.path.isfile(maskAddress3+'leftKidneyMaskCortex.nii.gz'):
            lkcor1= nib.load(maskAddress3+'leftKidneyMaskCortex.nii.gz');
            lkcor=lkcor1.get_data();#2*lkcor1.get_data()
        else:
            lkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress3+'rightKidneyMaskCortex.nii.gz'):
            rkcor1= nib.load(maskAddress3+'rightKidneyMaskCortex.nii.gz');
            rkcor=rkcor1.get_data()
        else:
           rkcor = np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii'):
            am1= nib.load(maskAddress+'aortaMask.nii');
            am=am1.get_data()  
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                lkm=lkmUpper+lkmLower;
            else:
                lkm=np.zeros(np.shape(am));
                
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                rkm=rkmUpper+rkmLower;
            else:
                rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;rkcor=0;lkcor=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()  

        # x=subjectInfo['timeRes'][patientName];
        x0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']       
        index = x0.index.values
        index = index[0]
        x = x0.loc[index]
        
        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);

        num2=seq2tres/seq1tres;
        im3=zoom(im2,(1,1,1,num2),order=0);
        im=np.concatenate((im1, im3), axis=3);
        
        
        if os.path.isfile(maskAddress2+'leftKidneyMaskCortex.nii.gz'):
            lkcor1= nib.load(maskAddress2+'leftKidneyMaskCortex.nii.gz');
            lkcor=lkcor1.get_data()#2*lkcor1.get_data()
        else:
            lkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
            
        if os.path.isfile(maskAddress2+'rightKidneyMaskCortex.nii.gz'):
            rkcor1= nib.load(maskAddress2+'rightKidneyMaskCortex.nii.gz');
            rkcor=rkcor1.get_data()
        else:
            rkcor=np.zeros([im.shape[0],im.shape[1],im.shape[2]]);
        
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                    lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                    lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                    lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                    lkm=lkmUpper+lkmLower;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                    rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                    rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                    rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                    rkm=rkmUpper+rkmLower;
            else:
                    rkm=np.zeros(np.shape(am));
        else:
            lkm=0;rkm=0;am=0;

    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        

    ### medulla
    lkmed = np.copy(lkm); # originally 2 and 0
    lkmed[lkcor == 2] = 0;
    lkmed[lkcor == 1] = 0;
    
    rkmed = np.copy(rkm); # originally 1 and 0
    rkmed[rkcor == 1] = 0;
    rkmed[rkcor == 2] = 0;

     
    oriKM = rkm+lkm;
    corOri = rkcor+lkcor;
    medOri = rkmed+lkmed;
    im=(im/np.amax(im))*100;
    
    return im, oriKM, boxes, rkm, lkm, lkmed, rkmed, medOri, lkcor, rkcor, corOri, seqNum


def writeMasksMulti(patientName,subjectInfo,reconMethod,Masks2Save,overwrite,medulla):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    

    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_T0.nii';
        img= nib.load(dataAddress)

        maskAddress='path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii') and not overwrite:    
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
            summationL = np.sum(Masks2SaveL)
            if summationL == 0.0:
                #Masks2SaveL = Masks2SaveL.astype('int');
                Masks2SaveL[112,112,16] = 1.0;
                
            summationR = np.sum(Masks2SaveR)
            if summationR == 0.0:
                #Masks2SaveR = Masks2SaveR.astype('int');
                Masks2SaveR[112,112,16] = 1.0;
                
                
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)
            
            if medulla == 1:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic_m.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic_m.nii.gz');
            else:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic_c.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic_c.nii.gz');
 
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_T0.nii';  
        img1= nib.load(dataAddress)

            
        maskAddress='path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii') and not overwrite:     
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
            summationL = np.sum(Masks2SaveL)
            if summationL == 0.0:
                #Masks2SaveL = Masks2SaveL.astype('int');
                Masks2SaveL[112,112,16] = 1.0;
                
            summationR = np.sum(Masks2SaveR)
            if summationR == 0.0:
                #Masks2SaveR = Masks2SaveR.astype('int');
                Masks2SaveR[112,112,16] = 1.0;
            
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img1.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img1.affine)
            
            if medulla == 1:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic_m.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic_m.nii.gz');
            else:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic_c.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic_c.nii.gz');
            
            oscommand='chmod -R 777 '+maskAddress;
            os.system(oscommand);

    return

def writeMasksMultiBaseline(patientName,subjectInfo,reconMethod,Masks2Save,overwrite,medulla):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    
    
    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_T0.nii';
        img= nib.load(dataAddress) 
        
        maskAddress='path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/'+patientName+'/';
        if os.path.isfile(maskAddress+'leftKidneyMask_baseline.nii') and not overwrite:    
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
            summationL = np.sum(Masks2SaveL)
            if summationL == 0.0:
                #Masks2SaveL = Masks2SaveL.astype('int');
                Masks2SaveL[112,112,16] = 1.0;
                
            summationR = np.sum(Masks2SaveR)
            if summationR == 0.0:
                #Masks2SaveR = Masks2SaveR.astype('int');
                Masks2SaveR[112,112,16] = 1.0;
                
                
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)
            
            if medulla == 1:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_baseline_m.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_baseline_m.nii.gz');
            else:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_baseline_c.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_baseline_c.nii.gz');
 
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_T0.nii';  
        img1= nib.load(dataAddress)
   
        maskAddress='path-to/folder-containing/medulla-cortex-segmentations/'+patientName+'_seq1/'+patientName+'/';
        if os.path.isfile(maskAddress+'leftKidneyMask_baseline.nii') and not overwrite:   
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
            summationL = np.sum(Masks2SaveL)
            if summationL == 0.0:
                #Masks2SaveL = Masks2SaveL.astype('int');
                Masks2SaveL[112,112,16] = 1.0;
                
            summationR = np.sum(Masks2SaveR)
            if summationR == 0.0:
                #Masks2SaveR = Masks2SaveR.astype('int');
                Masks2SaveR[112,112,16] = 1.0;
            
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img1.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img1.affine)
            
            if medulla == 1:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_baseline_m.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_baseline_m.nii.gz');
            else:
                nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_baseline_c.nii.gz');
                nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_baseline_c.nii.gz');
            
            
            oscommand='chmod -R 777 '+maskAddress;
            os.system(oscommand);
            
    
    return

def writeMasks(patientName,subjectInfo,reconMethod,Masks2Save,overwrite):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    
    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_T0.nii';
        img= nib.load(dataAddress)
        
        maskAddress='path-to-folder-containing-segmented-kidneys-in-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii') and not overwrite:   
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img.affine)
            nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic.nii.gz');
            nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic.nii.gz');    
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_T0.nii';  
        img1= nib.load(dataAddress)
        maskAddress='path-to-folder-containing-segmented-kidneys-in-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'leftKidneyMask_automatic.nii') and not overwrite:    
            print('Mask is already existant!');
        else:
            Masks2SaveR=Masks2Save['R'];Masks2SaveL=Masks2Save['L'];
            
            summationL = np.sum(Masks2SaveL)
            if summationL == 0.0:
                #Masks2SaveL = Masks2SaveL.astype('int');
                Masks2SaveL[112,112,16] = 1.0;
                
            summationR = np.sum(Masks2SaveR)
            if summationR == 0.0:
                #Masks2SaveR = Masks2SaveR.astype('int');
                Masks2SaveR[112,112,16] = 1.0;
            
            Masks2Save1R = nib.Nifti1Image(Masks2SaveR, img1.affine)
            Masks2Save1L = nib.Nifti1Image(Masks2SaveL, img1.affine)
            nib.save(Masks2Save1L,maskAddress+'leftKidneyMask_automatic.nii.gz');
            nib.save(Masks2Save1R,maskAddress+'rightKidneyMask_automatic.nii.gz');
            oscommand='chmod -R 777 '+maskAddress;
            os.system(oscommand);
    return


def baselineFinder(im):
    aortaPotentialTimesIM=im[75:150,:,0:20,0:50]; # keep first 15 dataPoints
    #aortaPotentialTimesIM=im[:,:,:,0:1];
    x=(aortaPotentialTimesIM>.8*np.max(aortaPotentialTimesIM)).nonzero()    
    
    x=(np.max(aortaPotentialTimesIM,axis=3)-np.min(aortaPotentialTimesIM,axis=3)>.6*np.max(aortaPotentialTimesIM)).nonzero()    
    #b = Counter(x[2]);
    #mostOccInZ=b.most_common(1)[0][0];
    
    medianOfValsInXaxisNdx=(abs(x[0]-np.median(x[0]))<10).nonzero()[0];
    medianOfValsInYaxisNdx=(abs(x[1]-np.median(x[1]))<10).nonzero()[0];
    medianOfValsInZaxisNdx=(abs(x[2]-np.median(x[2]))<10).nonzero()[0];
    commonXYconstraint=np.intersect1d(medianOfValsInXaxisNdx,medianOfValsInYaxisNdx);
    commonXYconstraint=np.intersect1d(medianOfValsInZaxisNdx,commonXYconstraint)
    allAortaPotentials=aortaPotentialTimesIM[x[0][commonXYconstraint],x[1][commonXYconstraint],x[2][commonXYconstraint],:]
    
    _,timeNdx = allAortaPotentials.max(1),allAortaPotentials.argmax(1)
    maxA=np.median(timeNdx)
    y=np.mean(allAortaPotentials[:,0:int(maxA)+1],0);
    y2=(y*400)/y.max();y3=y2-y2.min()+50;
    baseLine=(y3<0.5*(max(y3)-min(y3))).nonzero()[0][-1];

    return baseLine

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
#    
#    check4 = vol4D[:,:,:,jumpTimeSample+1:];  
    
    difference = fid[jumpTimeSample+1]-fid[jumpTimeSample];
    
    newVol4D = vol4D;
    oldVol4D = vol4D;

#    before = vol4D[:,:,1,jumpTimeSample:jumpTimeSample+1]
#    mB = np.mean(before);
#    after  = vol4D[:,:,1,jumpTimeSample:jumpTimeSample+1]-difference; 
#    mA = np.mean(after);
   
    newVol4D[:,:,:,jumpTimeSample+1:] = np.subtract(oldVol4D[:,:,:,jumpTimeSample+1:],difference)

    return newVol4D

def readData(patientName,subjectInfo,reconMethod,genBoundBox):
    
    # seqNum=subjectInfo['numSeq'][patientName];
    seqNum0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'numSeq']
    index = seqNum0.index.values
    index = index[0]
    seqNum = seqNum0.loc[index]

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'
    
    if reconMethod=='GRASP':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/method2/';
    elif reconMethod=='SCAN':
        dataAddress0='path-to-folder-containing-4D-volumes'+reconMethod+'/';


    if seqNum==1 or reconMethod=='GRASP':
        dataAddress=dataAddress0+patientName+'/recon'+reconMethod+'_4D.nii';
        img= nib.load(dataAddress)
        im=img.get_data()  
        
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'/';
        if os.path.isfile(maskAddress+'aortaMask.nii'):
            am1= nib.load(maskAddress+'aortaMask.nii');am=am1.get_data()  
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                lkm=lkmUpper+lkmLower;
            else:
                lkm=np.zeros(np.shape(am));
                
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                rkm=rkmUpper+rkmLower;
            else:
                rkm=np.zeros(np.shape(am));

        else:
            lkm=0;rkm=0;am=0;
        
        
    elif seqNum==2:
        dataAddress=dataAddress0+patientName+'_seq1/recon'+reconMethod+'_4D.nii';  
        img1= nib.load(dataAddress)
        im1=img1.get_data()  
        dataAddress=dataAddress0+patientName+'_seq2/recon'+reconMethod+'_4D.nii';  
        img= nib.load(dataAddress)
        im2=img.get_data()

        # x=subjectInfo['timeRes'][patientName];
        x0 = subjectInfo.loc[subjectInfo['Unnamed: 0'] == patientName, 'timeRes']       
        index = x0.index.values
        index = index[0]
        x = x0.loc[index]
        
        seq1tres=float(x.split("[")[1].split(",")[0]);
        seq2tres=float(x.split(",")[1].split("]")[0]);
        
        if seq2tres>seq1tres:
            #resample second to first
            num2=seq2tres/seq1tres;
            im3=zoom(im2,(1,1,1,num2),order=0);
            im=np.concatenate((im1, im3), axis=3);
        else:
            im=np.concatenate((im1, im2), axis=3);
            
        maskAddress='path-to-folder-containing-4D-volumes/'+patientName+'_seq1/';
        if os.path.isfile(maskAddress+'aortaMask.nii.gz'):
            am1= nib.load(maskAddress+'aortaMask.nii.gz');am=am1.get_data() 
            
            if os.path.isfile(maskAddress+'leftKidneyMask.nii.gz'):
                    lkm1= nib.load(maskAddress+'leftKidneyMask.nii.gz');lkm=2*lkm1.get_data()
            elif os.path.isfile(maskAddress+'leftKidneyMaskUpper.nii.gz'):
                    lkm1_upper= nib.load(maskAddress+'leftKidneyMaskUpper.nii.gz');lkmUpper=2*lkm1_upper.get_data() 
                    lkm1_lower= nib.load(maskAddress+'leftKidneyMaskLower.nii.gz');lkmLower=2*lkm1_lower.get_data() 
                    lkm=lkmUpper+lkmLower;
            else:
                    lkm=np.zeros(np.shape(am));
                    
            if os.path.isfile(maskAddress+'rightKidneyMask.nii.gz'):
                    rkm1= nib.load(maskAddress+'rightKidneyMask.nii.gz');rkm=rkm1.get_data()  
            elif os.path.isfile(maskAddress+'rightKidneyMaskUpper.nii.gz'):
                    rkm1_upper= nib.load(maskAddress+'rightKidneyMaskUpper.nii.gz');rkmUpper=2*rkm1_upper.get_data() 
                    rkm1_lower= nib.load(maskAddress+'rightKidneyMaskLower.nii.gz');rkmLower=2*rkm1_lower.get_data() 
                    rkm=rkmUpper+rkmLower;
            else:
                    rkm=np.zeros(np.shape(am));    
 
        else:
            lkm=0;rkm=0;am=0;

    
    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

        # bounding box center + width and height
        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    KM=np.uint16(rkm+lkm);
    im=(im/np.amax(im))*100;

    
    return im, KM, boxes, rkm, lkm



def baselineFinder2(patientName,subjectInfo):
    
    maskAddress='path-to-folder-containing-4D-volumes/'+patientName;
    path1 = maskAddress+'/aortaMask.nii.gz';
    path2 = maskAddress+'_seq1/aortaMask.nii.gz';
    
    if os.path.isfile(path1):
        am1= nib.load(path1);am=am1.get_data() 
    elif os.path.isfile(path2): 
        am1= nib.load(path2);am=am1.get_data() 
        
    im, _,_,_,_= readData(patientName,subjectInfo,'SCAN',0);
    im = computeNew4D(im);
    time = im.shape[3]
    
    aortaMask = np.zeros(np.shape(im));
    aortaInt = im;
    for xx in range (time):
        aortaMask[:,:,:,xx] = am;
    
    aortaInt[aortaMask==0]=0
    
    aortaMask = aortaInt
    
    xs,ys,zs,as1 = np.where(aortaMask!=0) 
    aortaMask2 = aortaMask[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1,min(as1):max(as1)+1]
    aortaMask2 = np.array(aortaMask2)
    aortaMask3 = np.reshape(aortaMask2,[aortaMask2.shape[0]*aortaMask2.shape[1]*aortaMask2.shape[2],time]);
    
    xs1,ys1 = np.where(aortaMask3 !=0) 
    aortaMask4 = aortaMask3[min(xs1):max(xs1)+1,min(ys1):max(ys1)+1]
    aortaMask4 = np.array(aortaMask4)
    
    whereIs = np.amax(aortaMask4, axis=1)
    
    indicesWhereIs = np.where(whereIs !=0)
    aortaMask4 = aortaMask4[indicesWhereIs, :]
    aortaMask4 = np.reshape(aortaMask4,[aortaMask4.shape[1],time])

    
    midWay = int(time/2)
    tWay = int(time/3)
    lookAtM = aortaMask4[:,1:midWay]
    lookAtT = aortaMask4[:,1:tWay]
    
    ranMax = np.amax(lookAtM, axis=1)   # Maxima along the second axis
    ranMin = np.amin(lookAtT, axis=1)   # Minima along the second axis
    rangeA = ranMax - ranMin
    
    
    maxKeepPercent = 40;
    condition = rangeA > (1-maxKeepPercent/100)*max(rangeA)
    valuesOfInterestINDICES = np.where(condition)
    
    aortaMask5 = aortaMask4[valuesOfInterestINDICES,:]
    aortaMask5 = np.reshape(aortaMask5,[aortaMask5.shape[1],time])
    

    checkLength = aortaMask5.shape[0];
    powerOfHF = []

    for xx in range(checkLength): 
        arrayVI = np.array(aortaMask5[xx,:]);
        maximumVI = max(arrayVI)
        peak, = np.where(arrayVI == maximumVI)
        peak = max(peak)
        diffS = np.diff(aortaMask5[xx,peak:],n=1,axis=0) 
        squaredDiffs = np.square(diffS);
        sumSquaredDiffs = np.sum(squaredDiffs)
        powerOfHF.append(sumSquaredDiffs)
        
    powerOfHF = np.array(powerOfHF)
    lenHF = powerOfHF.shape[0]
    
    sortedSmoothestVoxels = sorted(range(lenHF), key=lambda k:  powerOfHF[k])
    smoothestVoxels = sortedSmoothestVoxels[1:int(len(sortedSmoothestVoxels)/2)];
    selectedVoxels =  np.array(aortaMask5[smoothestVoxels,:]); 
    aif = np.median(selectedVoxels,axis=0); 
    
    aifMAX = max(aif)
    aifMndx, = np.where(aif == aifMAX)
    aifMndx = max(aifMndx)    
    aifDif = np.diff(aif[1:aifMndx-1],n=1,axis=0) 
    
    aifDif = np.array(aifDif.tolist())
    #ls = [i for i, e in enumerate(aifDif) if e != 0]
    baseLine2 = [index for index, item in enumerate(aifDif) if item != 0][-1] 
    
    if baseLine2 > 15:
       baseLine2 = 10; 

    
    return baseLine2 
