
from gurobipy import *
from re import compile
import random 
import numpy as np
from collections import Counter
from itertools import chain

images = []
labels = []
trainImageNames = []
testImages = []
testLabels = [] 
testImageNames = []
weightThr = 6000
pixelThr = 0.25
deltaWeight = 0.01
totalN = 5
globalWeights = []
accuracies = []
probThreshhold = 0.7
probThreshholdRev = 0.1
constr_set = False
vars_set = False
vars_q = []
constraints = []
cnames = []
    
def load_pgm_image(pgm):
    global xdim, ydim
    with open(pgm, 'rb') as f:
        f.readline()   # skip P5
        f.readline()   # skip the comment line
        f.readline().split()  # size of the image
        max_scale = int(f.readline().strip())
        endian = '>'
        image = np.fromfile(f, dtype = 'u1' if max_scale < 256 else endian+'u2')
        image = [int(str(x)) / float(max_scale) for x in image.tolist()]
        im = []
        for im1 in image:
            a = im1
            if a >= pixelThr:
                a = 1    
            else:
                a = 0
            im.append(a)
        image = im      
    return image

def readTrainingImages():
    global images, labels, pixelSize, trainImageNames

    with open('downgesture_train.list') as f:
        for training_image in f.readlines():
            training_image = training_image.strip()
            trainImageNames.append(training_image)
            images.append(load_pgm_image(training_image))
            if 'down' in training_image:
                labels.append(1)
            else:
                labels.append(0)
    
    pixelSize = len(images[0])

def readTestImages():
    global testImages, testLabels, testImageNames
    with open('downgesture_test.list') as f:
        for test_image in f.readlines():
            test_image = test_image.strip()
            testImageNames.append(test_image)
            testImages.append(load_pgm_image(test_image))
            if 'down' in test_image:
                testLabels.append(1)
            else:
                testLabels.append(0)
                
def getGraph():
    level1 = []
    level2 = [] 
    level3 = []
    vertexList = []
    weightList = []
    edgeList = []
    
    l1vertexCount = pixelSize
    thresholdProb = 0.6
        
    for i in range(0, l1vertexCount*totalN):
        
        vertex = 'v'+str(i)
        vertexList.append(vertex)
        weightList.append( abs(random.randint(0,4)) )
        
        if i<l1vertexCount:
            level1.append(vertex)
        else: 
            if i<l1vertexCount*(totalN - 1):
                level2.append(vertex)
            else:  level3.append(vertex) 
  
    # L1 connects to L2 AND L3          
    for v1l1 in level1:
        
        for v2l2 in level2:
            connProb= random.random()
            if connProb< thresholdProb: continue
            edgeList.append((v1l1,v2l2))
        
        for v3l3 in level3:
            connProb= random.random()
            if connProb< thresholdProb: continue
            edgeList.append((v1l1,v3l3))  
            
    # L2 connects to L3          
    for v2l2 in level2:
        
        for v3l3 in level3:
            connProb= random.random()
            if connProb< thresholdProb: continue
            edgeList.append((v2l2,v3l3))               
    
    return (vertexList, weightList, edgeList)

def trainModelWithGurobi():
    global var_names, weights, constraints, cnames, gloablWeights, accuracies, vars_set, constr_set, modelM, vars_q
    var_names, weights, edges = getGraph() 
    vars_p = []
    iters = 0
    choicearray = range(len(images))
    
    print "Training in progress"
    while iters < 500 :
        if iters > 0 and iters % 50 == 0:
            print "Inside training epoch:",iters
        imageruns = 0
        up_img = 0
        down_img = 0
        zero_labels = 0
        one_labels = 0
        vars_memory = []
        while imageruns < 50:
            k = np.random.choice(choicearray,replace = True)
            image = images[k]
            label = labels[k]
            if label == 1:
                one_labels +=1
            else:
                zero_labels +=1
            try:
                if not vars_set:
                    # Create a new model
                    modelM = Model("mip1")
                    i = 0
                    
                    for v, w in zip(var_names, weights):
                        vars_p.append(modelM.addVar(vtype = GRB.BINARY, name = v))
                        cnames.append('c' + v)
                    modelM.update()
                    vars_q = list(vars_p)
                    vars_set = True
                    
                for i in range(pixelSize):
                    vars_q[i] = image[i]
                     
                if not constr_set:
                    patternindex = compile(r"^[0-9+]$") #
                    for e in edges:
                        v1 = [x for x in e[0] if patternindex.match(x)]
                        v1 = int(''.join(v1))
                        if v1 < pixelSize:
                            v1 = vars_q[v1]
                        else:
                            v1 = vars_p[v1]
                        v2 = [x for x in e[1] if patternindex.match(x)]
                        v2 = int(''.join(v2))
                        if v2 < pixelSize:
                            v2 = vars_q[v2]
                        else:
                            v2 = vars_p[v2]
                        temp = [v1, v2]
                        constraints.append(quicksum(x for x in temp))
                    
                    for ceq, c in zip(constraints, cnames):
                        modelM.addConstr(ceq >= 1, c)    
                    constr_set = True
                
                sumv = quicksum(w * v2 for w, v2 in zip(weights, vars_q))
                modelM.setObjective(sumv, GRB.MINIMIZE)    
                modelM.setParam("OutputFlag",False)
                modelM.optimize()
                
                vars_update = [y.varName for y in modelM.getVars() if y.x == 1]
                img_index = [i for i, x in enumerate(image) if x == 1]
                img_data = [var_names[i] for i in img_index]
                vars_union = list(set(vars_update).union(set(img_data)))
                vars_memory.append(vars_union)
                if modelM.objVal <= weightThr and label == 1:
                    down_img += 1
                elif modelM.objVal > weightThr and label == 0:
                    up_img += 1
                imageruns+=1    
                
            except GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))
            
            except AttributeError:
                print('Encountered an attribute error')
            
        globalWeights.append(weights)
        accuracy = (up_img + down_img) / 50.0
        accuracies.append(accuracy)
        #print "total predicted right up, down", up_img, down_img,
        #print "Accuracy", accuracy
        vars_repeated = list(chain.from_iterable(vars_memory))
        repeated_count = dict(Counter(vars_repeated))
        if up_img / float(zero_labels) < down_img / float(one_labels):
            for p in repeated_count:
                if random.random() < probThreshhold: 
                    continue
                wIndex = var_names.index(p)
                weights[wIndex] += weights[wIndex] * int(repeated_count[p]) * deltaWeight
            
            for i in range(len(weights)):
                if random.random() > probThreshholdRev:
                    continue
                weights[i] -= weights[i] * deltaWeight
                
        else:
            for p in repeated_count:
                if random.random() < probThreshhold: 
                    continue
                wIndex = var_names.index(p)
                weights[wIndex] -= weights[wIndex] * int(repeated_count[p]) * deltaWeight
                
            for i in range(len(weights)):
                if random.random() > probThreshholdRev:
                    continue
                weights[i] += weights[i] * deltaWeight
                            
        iters+=1
    print "Training complete"
        
def runTestImagesOnModel():
    global modelM, vars_q
    print "\nPrediction of test images: \n"
    j = 0
    down = 0
    up = 0
    for imagetest,labeltest in zip(testImages,testLabels):
        try: 
            for i in range(pixelSize):
                vars_q[i] = imagetest[i]
                
            sumv = quicksum(w * v2 for w, v2 in zip(weights, vars_q))
            modelM.setObjective(sumv, GRB.MINIMIZE)            
            modelM.setParam("OutputFlag",False)
            modelM.optimize()
            
            print testImageNames[j],
            if modelM.objVal < weightThr: 
                print ", predicted down"
                if labeltest ==1:
                    down+=1
            else:
                print ", predicted 'not down'"
                if labeltest ==0:
                    up+=1
            j+=1
                
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            
        except AttributeError:
            print('Encountered an attribute error')
    
    #ones = [i for i, x in enumerate(testLabels) if x == 1]
    print
    #print "Total down images predicted right:",down, "(",len(ones),")"
    #print "Total 'not down' images predicted right:",up, "(", len(testImages) - len(ones),")"
    print "Accuracy of classifying the gestures in images:", round(((down + up) / float(len(testImages))) * 100,1), "%"

readTrainingImages()
readTestImages()
trainModelWithGurobi()
runTestImagesOnModel()