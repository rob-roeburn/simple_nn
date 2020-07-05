#!/usr/bin/env python
# coding: utf-8

import numpy as np
import h5py,sys

SIACOL  = ['Identifier','Age','sex','Diameter','BIN_DIAG','COLL_BIN','BG_BIN','BL_DISP','BL_BLUS','DER_MEL','DM_GLOB','ASYM','SYM1','SYM2','MEL_GLOB','DISP_BLUS','DIAM_6','NO_SYM_2','size','shape','colour','inflamed','bleed','sens','diam_7','Total','suspicious']
BIAS    = 0.02

def load_sia_data(filename):
    try:
        weights=[]
        weightfile = open("data.h5", "r")
        weightcontent = weightfile.read()
        for i, weightrow in enumerate(weightcontent.split('\n')):
            weights.append(weightrow)

        fn = open(filename, "r")
        contents = fn.read()
        for i, row in enumerate(contents.split('\n')):
            rowlist = row.split(',')
            trainbools = []
            clinical = []
            for j, element in enumerate(rowlist):
                if (4<j<18):
                    trainbools.append(element)
                if (17<j<26):
                    clinical.append(element)
                                                    # 0=ID, 1=age, 2=sex, 3=diameter - 4:  non bools
            print("ref  " + rowlist[0])             # 4=diagnosis                    - 1:  BIN_DIAG
            print("diag:")
            print(bool(int(rowlist[4])))
            print("\n")
            print("trainbools")
            print(trainbools)                       # 5-17                           - 13: training bools
            print("clinical")
            print(clinical)                         # 18-24                          - 7:  clinical observations, Total, suspicious

            for k,weightrow in enumerate(weights):
                if (len(weightrow)>0):
                    sum=0.0
                    for l, weight in enumerate(weightrow.split(',')):
                        if(l<len(trainbools)):
                            sum+=float(weight)+float(trainbools[l])
                        elif len(weight)>0:
                            print("Activation threshold:"+str(weight))
                            print("Sum:"+str(sum))
                            if bool(int(rowlist[4])) and not bool(float(sum)>float(weight)):
                                print("True real diag and False activated diag.  Add bias to weights.")
                                biasweights=''
                                for m,weight in enumerate(weightrow.split(',')):
                                    if (m<len(trainbools)):
                                        biasweights+=str((float(weight)+BIAS))+","
                                    else:
                                        biasweights+=weight
                                print("Original : "+weightrow)
                                print("Modified : "+biasweights)
                                weights[k]=biasweights
                            if not bool(int(rowlist[4])) and bool(float(sum)>float(weight)):
                                print("False real diag and True activated diag.  Subtract bias from weights.")
                                biasweights=''
                                for m,weight in enumerate(weightrow.split(',')):
                                    if (m<len(trainbools)):
                                        biasweights+=str((float(weight)-BIAS))+","
                                    else:
                                        biasweights+=weight
                                print("Original : "+weightrow)
                                print("Modified : "+biasweights)
                                weights[k]=biasweights
                            # if both are true or both or false, leave bias at 0
            print("\n\n")

    except Exception as e:
        print(e)
        print("Failed")
        return (None, None)
    return

def train(filename):
    load_sia_data(filename)

train(sys.argv[1])
