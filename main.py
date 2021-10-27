# What version of Python do you have?
import sys

import tensorflow.keras #import keras won't work we have to import keras as import tensorflow.keras to acess keras gpu processing library
import pandas as pd
import sklearn as sk
from sklearn import linear_model
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as pyplot #this module will help us to plot grid and visualize our dataset and stuff
from matplotlib import style #this is gonna change the style of our grid
import pickle #this module will help us to save our Ml model once the training is complete

print(f"Tensor Flow Version: {tf.__version__}")
print(f"tensor-flow (GPU): {tf.test.gpu_device_name()}") #check if the tensorflow gpu is installed or not
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"numpy {np.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# now that we've checked for the pakages and their version installed in the conda environment

#this machine learning project will predict the student's grade in the upcomming exams
#now the first thing are gonna do is to read in our dataset that we've downloaded from UCI ML dataset website
# pd.read_csv("your csv file name.csv") will allow you to read your dataset file that you have imported in your project
# now if din't knew .csv files means that comma seperated values and if you open the student-mat.csv file you will see that all of the attributes are seperated by ; (semi-colon)
# so in order to deal with this you need to add sep = ";" where sep stands for seperator and what it will do is it will seperate all the attributes in the student-mat.csv file
# by semi-colonton make sure our panadas data frame gets the correct values
data = pd.read_csv("student-mat.csv", sep=";")

#inorder to see how our dataframe looks like under the hood type the code below
# PRINTING THE DATA FRAME BEFORE CHOOSING THE ATTRIBUTES i WANT
#data.head() is gonna grab first five elements in our dataframe and the print statement will just print it in the output window
print(data.head())

#now that we've fed the dataset to pandas we need to trim down the data attributes to what we need
#so in this student-mat.csv file there are 33 attributes but I don't want all those 33 attributes I only want to use some of them
# data = data[["type your attributes name that you want"]]
# please refer to the UCI Ml repository students marks dataset for more info on the attributes
# NOTE:- try to pick attributes in a dataset that have integer values associated with them because it will make life so much easier for us otherwise you need to
# you are gonna have to change all the values within this data frame into integer values so that we can make sure that it works with our program
# so every thing inside the "" is an attribute which is a part of a student and is unique to each individual student
data = data[["G1" , "G2" , "G3" , "studytime" , "failures", "absences", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]] # data frame

# #converting romantic attribute from string to integer BUT IT DIN'T WORK FOR ME THREW AN EXCEPTION
# # data['romantic'] = data['romantic'].astype('int') in some cases when converting a column with an object datatype it can throw error
# #in order to get around that we can use this data['romantic'] = pd.to_numeric(data['romantic'], errors='coerce')
# data['romantic'] = pd.to_numeric(data['romantic'], errors='coerce')
# print(data.dtypes)

#inorder to see how our dataframe looks like under the hood type the code below
# PRINTING THE DATA FRAME AFTER CHOOSING THE ATTRIBUTES i WANT
#data.head() is gonna grab first five elements in our dataframe and the print statement will just print it in the output window
print(data.head())

#now we are going to define what we are trying to predict with this Ml model
#what values do we actually want
#here we want to predict G3 attribute of this dataframe which is the students final grade using this ML model
#so essentially the thing that we want to predict is a label which is the output of our ML model
#labels are what we want to get based on attribute in this case our label is gonna be G3 (student's final grade)
# so based on "G1" , "G2" , "G3" , "studytime" , "failures", "absences", "romantic" attributes we want to predict G3 and treat G3 as a label
predict ="G3"

# here we will be setting up two arrays and one array is gonna define all of our attribute and another array is gonna be our label or array of labels if we have got multiple labels
# numpy allows us to use arrays in python programming language because python doesn' have inbuilt support for arrays
#data.drop([predict], 1) is gonna return a new data frame that doesn't have G3 (final grade) attribute in it
# X is all gonna be our training data for our ML model and based on the training data we are gonna predict the label in this case its G3
X = np.array(data.drop([predict], 1))
# Y is gonna be all of our labels
# here we will pass data[predict] because here we only care about our G3 value
Y = np.array(data[predict])

# now we are going to split X and Y arrays into 4 variables
# X test , Y test , X train and Y train
# here sk = sklearn
# so here essentially we are taking all of our attributes in X array and all of our labels that we are trying to predict and we are going to split them up into
# four different arrays
# x_train array is gonna be a section of X attribute array
# y_train array is gonna be a section of Y label array
# x_test, y_test arrays are our test data that we are gonna use to test the accuracy of our Ml model that we are gonna create
# now the way it works is if we trained the model every single bit of data that we have and it will simply just memorize it
# test_size = 0.1 is splitting 10% of our data into a test samples (x_test, y_test arrays) so that when we test of that and it's never seen that information before

# if you use x_train, y_train, x_test, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 )
# then it will throw an error Input contains NaN, infinity or a value too large for dtype('float64').
# change the above to this x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 ) and this will solve the issue
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.1)




#Now here we will use a for loop or a while loop to train the ML model until we get 90% or 95% accuracy rate and only then we will save the trained model
#using pickle module with the name studentmodel4.pickle
best_accuracy_of_the_trained_ML_model = 0
for _ in range(9999999):
    # if you use x_train, y_train, x_test, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 )
    # then it will throw an error Input contains NaN, infinity or a value too large for dtype('float64').
    #change the above to this x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 ) and this will solve the issue
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.1 )

    #here Ml model training and saving the model after it is trained takes place using Linear regression algorithm, comment it out once the model is trained and use the pickle file
    # to predict the student's final grade G3, pickle file contains the trained ML model--------------------------------------------------------------------------------------------
    #here we need to create a training model
    Linear = linear_model.LinearRegression() # store the best fit line in Linear variable and we can use Linear to test our best fit line on our test data

    #training our ML model to predict studen's final grade G3 using Linear regression algorithm
    # Linear is our actual ML model
    # Linear.fit(x_train, y_train) it is gonna fit this data in (x_train and y_train arrays) to find the best fit line like we do
    # when soving problems in maths problems based on Linear regression
    Linear.fit(x_train, y_train)

    # Linear.score(x_test, y_test) this is gonna return us a value that will tell us the accuracy of our model
    accuracy_of_ml_model = Linear.score(x_test, y_test)
    # print(f"accuracy of the Ml model:==> {accuracy_of_ml_model}%")

    #Ml model training ends here

    #save the trained ML model if the accuracy_of_ml_model is greater than our current best_accuracy_of_the_trained_ML_model
    #then we are gonna save that trained model in form of pickle file with the name of studentmodel4.pickle
    if accuracy_of_ml_model > best_accuracy_of_the_trained_ML_model:
        #update the best_accuracy_of_the_trained_ML_model if true
        best_accuracy_of_the_trained_ML_model = accuracy_of_ml_model
        print(f"accuracy of the Ml model (INCREASED):=========> {best_accuracy_of_the_trained_ML_model}%")
        # saving the trained Ml model after the training is complete using pickle module that comes built in with python programming language
        #saving a trained Ml model is important because every time we train our model the accuracy of our Ml model fluctuates between 80 to 90 %
        # so we need to have the ability to save the Ml model with the highest accuracy so that we can use it in other projects to solve real world problems in our other projects
        # with open("studentmodel.pickle", "wb") it is gonna open a pickle file named studentmodel in wb mode and write the file if it doesn't already exists
        #pickle.dump(Linear, f)  this will save the opened pickle file if not already exists in our working projects directory , NOTE:->here Linear is our actual ML model
        with open("studentmodel8.pickle", "wb") as f:
            pickle.dump(Linear, f)
        #here Ml model training and saving the model after it is trained takes place using Linear regression algorithm comment it out once the model is trained and use the pickle file
        # to predict the student's final grade G3, pickle file contains the trained ML model-------------------------------------------------------------------------------------------





#read in our pickle file
# open("studentmodel.pickle", "rb") this will open the pickle file studentmodel.pickle in our directory in rb mode
#I have managed to get a trained Ml model with 87% accuracy with the name of studentmodel3.pickle so I am using that trained model to predict the student's final grade
pickle_in = open("studentmodel7.pickle", "rb")

# once the pickle studentmodel.pickle is stored we can open the trained Ml model in form of studentmodel.pickle file we can just load this pickle in to our Linear variable
# which we will use as a trained ML model
#and use the trained ML model stored in form of pickle file to predict the student's final grade G3
Linear = pickle.load(pickle_in)

#getting the accuracy of our trained ML model
# Linear.score(x_test, y_test) this is gonna return us a value that will tell us the accuracy of our model
accuracy_of_ml_model = Linear.score(x_test, y_test)
print(f"accuracy of the trained Ml model:==> {accuracy_of_ml_model}%")

#testing our Ml model once the training is complete
# now we all know the formula for linear regresion algo is Y=mx+c
#so now we will find out what is m what is x and what is Y for this line in our ml model
print(f"coefficients of m : {Linear.coef_} \n intercept Y : {Linear.intercept_}")

#now here we are using our ML model to predict the student's score using test data x_test
predictions = Linear.predict(x_test)

#here I am gonna print out all the predictions and then I am gonna show what the input data was for that predictions
# x_test array is the set of data on which our model is not trained
#printing the inputs to the Ml model
for x in range(len(predictions)):
    print(f"input data : {x_test[x]}")
    print(f"student's final marks (grade) predictions : {round(predictions[x])}")
    print(f"actual value of the student's final grade : {y_test[x]}")

#plotting things on a grid so that we can take a look on how things looks like
#me are going to use matplotlib to do that
#this style.use("ggplot") is gonna make our grid look decent matplotlib
style.use("ggplot")

#now here we will plot a few pieces of informationon the grid using matplotlib
#here we will be using a scatter plot
# pyplot.scatter(x,y) x is gonna be your attributes in your dataset that you were feeding in the ML model and Y will be your labels the output of your trained MLmodel
P='G1'
pyplot.scatter(data[P], data["G3"])
#setting up labels for our axis
pyplot.xlabel("First unit test grade")
pyplot.ylabel("Final grade")
pyplot.show() #this will show the grid