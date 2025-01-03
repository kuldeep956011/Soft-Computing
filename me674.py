import math, random, sys       #using required modules
sys.setrecursionlimit(10000)
#########required functions

#############################defining element wise multiplication of 2 vectors/list or DOT product (must be of EQUAL LENGTH)
def vecMult(a,b):
    n=len(a)    #n=len(a)=len(b)
    dotSum = 0
    for i in range(0,n):
        dotSum = dotSum + a[i]*b[i]
    return dotSum

#writing functions to do matrix multiplication & addition
def matMult(A,B):       #doing A*B  (columns in A  = rows in B )
    colA= len(A[0])       #columns in A
    rowB=len(B)            #rows in B
    result = []
    if (colA==rowB):
        
        for i in range(len(A)):
            
            newRowElement=[]
            for j in range(len(B[0])):     #column counter
                sum=0
                for k in range(len(B)):
                    sum += A[i][k]*B[k][j]
                newRowElement.append(round(sum,5))
            result.append(newRowElement)
        #print("------------------matrix multiplication----------------")
        return result       
            

        #print("multiplication is possible")
    else:
        print("multiplication isn't possible")

#print(matMult([[1,2,3],[4,5,6],[7,8,9]],[[3,2],[1,2],[4,3]]))

#######finding sum of 2 matrices#############
def matSum(A,B):      #order(A)=order(B)
    rowA= len(A)           #rows in A
    colA=len(A[0])         # columns in A
    rowB=len(B)             #rows in B
    colB= len(B[0])           #columns in B
    if ((rowA==rowB) and (colA==colB)):
        matrix=[]
        for i in range(rowA):
            matrixRowElement=[]
            for j in range(colA):
                sum = (A[i][j]+B[i][j])
                matrixRowElement.append(round(sum,5))
            matrix.append(matrixRowElement)
        #print("-------matrices can be added---------")
        return matrix
    else:
        print("can't add matrices")
#print(matSum([[-1,3]],[[3,4]]))
############################################################

##########assuming our transfer functioin is log-sigmoid#################   
def transferFunction(a,x):
    return 1/(1+math.exp(-a*x))

####################################################################

def L2Norm(a):           #will find length of vector
    n = len(a)
    sum =0
    for i in range(n):
        sum = sum + a[i]*a[i]
    return round(math.sqrt(sum),5)



################reading data from input text file##################
file = open("normalisedSirData.txt",'r')  
inputPattern=[]            #2D list to store the patterns
count = 0                        #counter to count total no. of patterns
firstLine = file.readline()             #type string

while firstLine != "":
    strPattern=firstLine.split()            #each pattern will be string       
    numPatttern=[]                             #use to store numeric pattern rather than string pattern
    for i in strPattern:
        numPatttern.append(float(i))                #type conversion

    count += 1
    #print(count)
    #print(numPatttern)
    inputPattern.append(numPatttern)
    firstLine = file.readline()
#print("Out")
#print(f"Total Pattern are = {count}")

####################################################################




#####################generating matric of size (r,c) r= no. of rows, c= no. of columns  connection weight & bias matrices############
parameters='True'               #to check whether user gives Correct inputs or not
while (parameters == 'True'):
    n1=int(input("Enter number of input neurons: "))          #no. of input neurons
    n2=int(input("Enter number of output neurons: "))         #no. of output neurons
    n3=int(input("Enter number of training patterns: "))    # no. of training patterns
    n5=int(input("Enter number of Hidden layers: "))            #no. of hidden layers
    n6= int(input("Enter number of Hidden neurons: "))           # no. of neurons in each hidden layer
    n4=count-n3     #no. of testing patterns= total input  - number of training patterns
    if (n5!=0 and n6!=0):   
        #print("-----------------connection weight dictionary------------------------------")
        cln= n6     #current layer neurons
        pln= n1    #previous layer neurons
        dictionaryCW={}  # to  store connection weight matrices

        ##############making first/initial guess connection weight matrix, b/w input & first hidden layers############
        firstConnectionweight= []   # store connection weight values
        for i in range(0,n6):
            intermediate = []
            for j in range(0,n1):
                ele=  random.uniform(-1,1)                          #random number b/w (-1,1)              
                intermediate.append(round(ele,5))
            firstConnectionweight.append(intermediate)

        dictionaryCW[1]=firstConnectionweight
        ###############connection weight matrices from other layers###############
        
        for k in range(2,n5+1):
            connectionweight= []   # store connection weight values
            for i in range(0,n6):
                CWintermediate = []
                for j in range(0,n6):
                    ele= random.uniform(-1,1)
                    CWintermediate.append(round(ele,5))
                connectionweight.append(CWintermediate)

            dictionaryCW[k]=connectionweight
        ###############connection weight matrices b/w last hidden & output layers###############
        cw=[]
        for i in range(n2):
            interList=[]
            for j in range(n6):
                number=  random.uniform(-1,1)    #initial connection weight between (-1,1)
                interList.append(round(number,5))
            cw.append(interList)
        dictionaryCW[n5+1]=cw

        #print(dictionaryCW)                           ###########initial connection weight dictionary
        parameters= 'False'
        
    elif(n5==0 and n6== 0):    #no hidden layer in NN
        print("only 2 layers")
        dictionaryCW={}  # to  store connection weight matrices
        mainList=[]
        for i in range(n2):
            list2=[]
            for j in range(n1):
                num=  random.uniform(-1,1)
                list2.append(round(num,5))
            mainList.append(list2)
        dictionaryCW[1]=mainList 
        #print(dictionaryCW)  
        parameters= 'False'
    else:       #either hidden layer=0 or hidden neuron = 0
        print("Invalid combination of hidden layer & hidden neurons, Try again")
        parameters= 'True'
#print("I am out")

##################generating bias matrices########################################
#print("bias values")
biasVal=[]                         
for i in range(0,n5+1):
    val=  random.uniform(-1,1)
    biasVal.append(round(val,5))
#print(biasVal)

dictionaryB={}
for k in range(0,n5):
    bias=[]
    for i in range(0,n6):
        interBias=[]
        for j in range(0,n3):
            element= biasVal[k]
            interBias.append(element)
        bias.append(interBias)
    dictionaryB[k+1]=bias
#print("incomplete bias dictionary")

#print(dictionaryB)
####################making last bias matrix, b/w last hidden layer & output layer#############
lastBiasMatrix=[]
for k in range(1,n2+1):
    INTERMEDIATE=[]
    for i in range(0,n3):
        ELEMENT=biasVal[len(biasVal)-1]
        INTERMEDIATE.append(ELEMENT)
    lastBiasMatrix.append(INTERMEDIATE)
dictionaryB[len(dictionaryB)+1]=lastBiasMatrix

#print("last bias matrix")
#print(lastBiasMatrix)
# print("--------------------final bias dictionary-----------------------")
# print(dictionaryB)             #########bias dictionary




###############forming list to store output values of the pattern###############
#print(inputPattern)
outputPattern=[]
for i in range(count):           #count = totoal number of patterns
    val=[]
    for k in range(0,n2):
        listOfOutputs=inputPattern[i][n1+k]      #in case we have more than 1 outputs
        val.append(listOfOutputs)      
    outputPattern.append(val)
output=[outputPattern]       #list storing ALL output neuron values/target values
#print("------output matrix---------")
#print(output)                
#print(len(output),len(output[0]))


#removing output values from inputPattern list

for i in range(count):
    inputPattern[i].pop()
#print(inputPattern)
#print(len(inputPattern),len(inputPattern[0]))


patternMatrix=list(zip(*inputPattern))     #finding transpose of inputPattern list,useful in doing Matrix multiplication
#print(patternMatrix)
#print(len(patternMatrix),len(patternMatrix[0]))



#print(dictionaryCW[1])
#print(dictionaryB[1])

#product_of_connectionWeight_and_inputPattern=matMult(dictionaryCW[1],patternMatrix)
#print(product_of_connectionWeight_and_inputPattern)
#print(len(product_of_connectionWeight_and_inputPattern),len(product_of_connectionWeight_and_inputPattern[0]))


#####################################splitting pattern matrix & output value of pattern into training & testing matrices

training_input_pattern_matrix=[]
training_output_pattern_matrix=[]
training_input_pattern_matrix=[]
for i in range(n1):
    rowInput=[]
    
    for j in range(n3):
        rowInput.append(patternMatrix[i][j])
        
        
    training_input_pattern_matrix.append(rowInput)
#print("---------training_input_pattern_matrix------------------")
#print(training_input_pattern_matrix) 
#print(len(training_input_pattern_matrix),len(training_input_pattern_matrix[0]))
#print(patternMatrix)


#####output/target values corresponding to input training patterns######
rowOutput=[]
for i in range(n3):
    rowOutput.append(output[0][i])
training_output_pattern_matrix.append(rowOutput)
# print("---------------training_output_pattern_matrix-----------------------")
# print(training_output_pattern_matrix)
###################################################
testing_input_pattern_matrix=[]
testing_output_pattern_matrix=[]
for i in range(n1):
    rowwInput=[]
    for j in range(n3,count):                  #taking testing input patterns immediate after the training input patterns
        rowwInput.append(patternMatrix[i][j])
    testing_input_pattern_matrix.append(rowwInput)
#print("---------------------testing_input_pattern_matrix------------------------")
#print(testing_input_pattern_matrix)
#print(len(testing_input_pattern_matrix),len(testing_input_pattern_matrix[0]))


#####output patterns corresponding to testing patterns######
rowwOutput=[]
for j in range(n3,count):
    rowwOutput.append(output[0][j])
    #print(rowwOutput)
testing_output_pattern_matrix.append(rowwOutput)
#print("---------------testing_output_pattern_matrix-----------------------")
#print(testing_output_pattern_matrix)


#######writing input to jth neuron of 1St hidden layer for pth pattern
# Ihj = matSum(matMult(dictionaryCW[1],training_input_pattern_matrix), dictionaryB[1])
# print("-------------Input to jth neuron of 1st hidden layer for pt pattern----------------------")
# print(Ihj)





######################actual calculation begins here###################
check='true'
toleranceList=[]
iterationList=[]
iteration=1
max_iteration=10000
while(iteration <= max_iteration):
    iterationList.append(iteration)
    # print(f"iteration no= {iteration}")
    # print(f"dictionary[1] = {dictionaryCW[1]} dictionary[2] = {dictionaryCW[2]}")
    ######writing output of jth neuron of 1st hidden layer for pth pattern
    copy=training_input_pattern_matrix  #used to update training_input_pattern_matrix
    # print("---------Copy Matrix---------------- ")
    # print(copy)
    outputMatrix = []                         #use to store all the output from all the hidden and output layers
    for i in range(1,n5+2):
        inputMatrixElement = matSum(matMult(dictionaryCW[i],copy), dictionaryB[i])
        #print(f"Input Matrix Element = {inputMatrixElement}")
        outputMatrixElement = []
        for j in range(len(inputMatrixElement)):
            ROW=[]
            for k in range(n3):
                # print(f"j={j} k={k}")
                # print(inputMatrixElement[j][k])
                value=transferFunction(1,inputMatrixElement[j][k])       #a=1, tranfer function coefficient
                ROW.append(round(value,5))
            outputMatrixElement.append(ROW)
            #print(f"OutputMatrixElement = {outputMatrixElement}")
        outputMatrix.append(outputMatrixElement)
        
        copy=outputMatrixElement          #input to next layer is output of previous layer
        # print("-----------updated copy matrix--------------")
        # print(f"updated copy = {copy}")
    # print("---------------Output Matrix including each hidden and output layers------------------------")
    # print(outputMatrix)
    predictedOutput = outputMatrix[len(outputMatrix)-1]      #consist of only output neurons values
    # print("------------------predicted output neuron values-----------------")
    # print(f"predicted output:{predictedOutput}")

    #####################defining error of k-th output layer neuron############  assuming 3 layer network (only 1 hidden layer alomg with input and output layers)



    targetOutput =[]
    for i in range(0,n3):
        targetOutput.append(training_output_pattern_matrix[0][i])
    targetOutputMatrix = list(zip(*targetOutput))
    #print(f"targetOutputMatrix = {targetOutputMatrix}")


    errorMatrix = []        # will be a matrix of size n3*p (output neuron * patterns)
    for i in range(n2):
        errorRow=[]
        for j in range(n3):
            
            #print(f"training_output_pattern_matrix[0][0][{i}] = {training_output_pattern_matrix[0][0][i]}, predictedOutput[0][{i}] = {predictedOutput[i][0]}")
            errorElement = 0.5*((targetOutputMatrix[i][j] - predictedOutput[i][j]))**2
            #print(f"errorElement = {errorElement}")           
            errorRow.append(round(errorElement,5))       #error row will have error of k-th output neuron for all input training patterns 
            #print(f"---------Error of {i+1}-th output neuron for {j+1}-th pattern--------------- ")            
            #print(f"errorRow  = {errorRow}")    
        errorMatrix.append(errorRow)
    # print("-------------------Error Matrix------------------------")
    # print(f"Error Matrix = {errorMatrix}")
    #print(f"len(targetOutputMatrix ={len(targetOutputMatrix)}, len(predictedOutput) = {len(predictedOutput)}")

    #######finding MSE(MEAN SQUARE ERROR) for Batch Mode of training###################
    mseList=[]

    for i in range(0,len(errorMatrix)):
        sum = 0
        for j in range(0,len(errorMatrix[0])):
            sum += errorMatrix[i][j]
        mseElement = sum / n3
        mseList.append(round(mseElement,5))
    #print(f"MSE matrix = {mseList}")

    ######## Tolerance value/converging criterion########
    
    tolerance = L2Norm(mseList)
    toleranceList.append(tolerance)
    #print(f"tolerenceList = {toleranceList}")
    #print(f"tolerance = {tolerance}")

    if (tolerance < 0.0001):
        #print("I am in if")
        #toleranceList.append(tolerance)
        print(f"---------------------optimal connection weights matrices are -----------------------")
        print(f"V_ij ={dictionaryCW[1]}   W_jk = {dictionaryCW[2]}")
        break
        
        
        
    else:
        #print("I am in else")


    ##########updating connection weights between output and hidden layer############

        learningRate=0.6     # eta
        
        #######Output of each neuron of the hidden layer = outputMatrix[0]###################
        hiddenOutputMatrix=outputMatrix[0]
        #print(f"hiddenOutputMatrix = {hiddenOutputMatrix}")


        ############w_jk(new) = w_jk(old) + deltaW   all are of order n2*n6 ################

        b=1  #transferfunction coeficient
        bigList=[]           #use to find deltaW (see the expression of deltaW (T_ok - O_Ok)*b*O_ok*(1-O_ok))
        for j in range(0,n2):
            randomList=[]
            for p in range(0,n3):
                randomListElement = (targetOutputMatrix[j][p] - predictedOutput[j][p])*b*predictedOutput[j][p]*(1-predictedOutput[j][p])
                randomList.append(round(randomListElement,5))
            bigList.append(randomList)
        #print(f"bigList = {bigList}")

        ###############finding deltaW
        deltaW=[]
        for i in range(0,n2):
            deltaWrow=[]
            for j in range(0,n6):
                deltaWrowElement = (learningRate*(vecMult(bigList[i],hiddenOutputMatrix[j]))) / (n3)   #doing (T_ok - O_Ok)*b*O_ok*(1-O_ok) * (O_hj)
                deltaWrow.append(round(deltaWrowElement,5))
            deltaW.append(deltaWrow)
        #print(f"delta W = {deltaW}")


        ################finding UPDATED VALUE of W_jk##################  w_jk(new) = w_jk(old) + delta w ############  all are of order = n2*n6
        ##Assuming only 1 HIDDEN LAYER
        W_old = dictionaryCW[2]          #2nd element of connection weight dictionary i.e. dictionaryCW[2]
        #print(f"W_old = {W_old}")


        W_new = matSum(W_old,deltaW)
        
        #print(f"W_new = {W_new}")

        ############################doing BACK PROPAGATION###########################
        #error of each hidden neuron = average error of all output neurons for all patterns

        deltaV = [[0 for _ in range(n1)] for _ in range(n6)]
        # Ensure that dimensions are correct
        # print(f"Shape of W_new: {len(W_new)}x{len(W_new[0])}")
        # print(f"Shape of training_input_pattern_matrix: {len(training_input_pattern_matrix)}x{len(training_input_pattern_matrix[0])}")

        # Ensure valid indices for all matrices
        for i in range(n6):
            for j in range(n1):
                summation = 0
                for k in range(n2):
                    for p in range(n3):
                        term1 = (targetOutputMatrix[k][p] - predictedOutput[k][p])
                        term2 = b**2 * predictedOutput[k][p] * (1 - predictedOutput[k][p])

                        # Check if the indices are valid for hiddenOutputMatrix
                        if j < len(hiddenOutputMatrix) and p < len(hiddenOutputMatrix[j]):
                            term3 = hiddenOutputMatrix[j][p] * (1 - hiddenOutputMatrix[j][p])
                        else:
                            term3 = 0  # Default to 0 if out of bounds

                        # Calculate term4 as a weighted sum
                        # term4 = 0
                        # for idx, value in enumerate(training_input_pattern_matrix[i]):
                        #     if k < len(W_new) and j < len(W_new[k]):
                        #         term4 += W_new[k][j] * value  # Ensure valid indices for W_new
                        #     else:
                                
                        #         #print(f"Invalid index for W_new at k={k}, j={j}")
                        #         term4 += 0  # Default to 0 if out of bounds

                        # Accumulate the summation
                        summation += term1 * term2 * term3              #* term4

                # Update deltaV
                deltaV[i][j] = round(((learningRate / (n2 * n3)) * summation),5)




        # deltaV now holds the computed values as per the equation
        #print("Updated deltaV:", deltaV)

        ##########finding updated V_ij = V_ij(old) + deltaV
        V_old = dictionaryCW[1]
        V_new = matSum(V_old,deltaV)
        #print(f"V_new = {V_new}")
        dictionaryCW[1]=V_new
        dictionaryCW[2]=W_new
        #print(f"dictionary[1] = {dictionaryCW[1]} dictionary[2] = {dictionaryCW[2]}")
        # toleranceList.append(tolerance)
        # print(f"toleranceList = {toleranceList}")
        iteration += 1

print(f"---------------------optimal connection weights matrices are -----------------------")
print(f"V_ij ={dictionaryCW[1]}   W_jk = {dictionaryCW[2]}")
# print(f"tolerenceList = {toleranceList}")
# print(f"iterationList = {iterationList}")


with open('output.txt', 'w') as file:
    # Iterate through the lists and write to the file
    for item1, item2 in zip(iterationList, toleranceList):
        file.write(f'{item1}, {item2}\n')
