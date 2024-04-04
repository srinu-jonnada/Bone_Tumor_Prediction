from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import imutils
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
import ftplib
from tkinter import ttk
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold

main = tkinter.Tk()
main.title("Identifying Bone Tumor using X-Ray Images") #designing main screen
main.geometry("1300x1200")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected','Tumor Detected']

with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
json_file.close()    
segmented_model.load_weights("Model/segmented_weights.h5")
segmented_model._make_predict_function()

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result    

def tumorSegmentation(filename):
    global segmented_model
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",segmented_image*255)
    edge_detection = edgeDetection()
    return segmented_image*255, edge_detection
    

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename+"/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/no/"+name,0) #reading images
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) #processing and normalization images
                img = cv2.resize(img, (128,128)) #resizing images
                im2arr = np.array(img) #extract features from images
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(0)
                print(filename+"/no/"+name)

        for root, dirs, directory in os.walk(filename+"/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/yes/"+name,0)
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(1)
                print(filename+"/yes/"+name)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    print(X.shape)
    print(Y.shape)
    print(Y)
    cv2.imshow('ss',X[20])
    cv2.waitKey(0)
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(Y)))+"\n\n")
    text.insert(END,"Class labels found in dataset : "+str(disease))
     
def trainTumorDetectionModel():
    global accuracy
    global classifier
    
    YY = to_categorical(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")
        classifier._make_predict_function()           
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        classifier = Sequential() 
        classifier.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 1), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim=128, activation='relu'))
        classifier.add(Dense(output_dim=2, activation='softmax'))
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()

    f = open('Model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[-1] * 100  # Use the last accuracy value
    text.insert(END, '\n\nCNN Bone Tumor Model Generated.\n\n')
    text.insert(END, "CNN Bone Tumor Prediction Accuracy on Test Images: {:.2f}%\n".format(accuracy))

def create_svm_model():
    svm_model = SVC(kernel='linear', C=1)  # You can customize kernel and C parameters
    return svm_model

def tumorClassification():
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename,0)
    img = cv2.resize(img, (128,128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,128,128,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    if cls == 0:
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        cv2.putText(img, 'Classification Result : '+disease[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : '+disease[cls], img)
        cv2.waitKey(0)
    if cls == 1:
        segmented_image, edge_image = tumorSegmentation(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        cv2.putText(img, 'Classification Result : '+disease[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : '+disease[cls], img)
        cv2.imshow("Tumor Segmented Image",segmented_image)
        cv2.imshow("Edge Detected Image",edge_image)
        cv2.waitKey(0)

def trainSVMDetectionModel():
    global accuracy
    global svm_model
    global X
    global Y
    global text

    # Encode labels if not already encoded
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Convert labels to categorical format
    YY = to_categorical(Y_encoded)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, YY, test_size=0.2, random_state=42)

    # Reshape x_train and x_test
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)

    svm_model = create_svm_model()
    svm_model.fit(x_train_flattened, np.argmax(y_train, axis=1))

    # Evaluate the SVM model on the test set
    accuracy = svm_model.score(x_test_flattened, np.argmax(y_test, axis=1)) * 100
    
    text.insert(END, '\n\nSVM Bone Tumor Model Generated.\n\n')
    text.insert(END, "SVM Bone Tumor Prediction Accuracy on Test Images: {:.2f}%\n".format(accuracy))


def compareAlgorithms():
    global svm_model
    global classifier
    global X
    global Y
    global accuracy

    # Train SVM model and get accuracy
    svm_accuracy = trainSVMDetectionModel()

    # Train CNN model
    cnn_accuracy = trainTumorDetectionModel()

    # Create a count plot
    models = ['SVM', 'CNN']
    accuracies = [svm_accuracy, cnn_accuracy]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('SVM vs. CNN Bone Tumor Detection Model Comparison')
    plt.ylim(0, 100)  # Adjust ylim if needed
    plt.show()


def graph():
    f = open('Model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', label='Loss')  # Remove color='red'
    plt.plot(accuracy, 'ro-', label='Accuracy')  # Remove color='green'
    plt.legend(loc='upper left')
    plt.title('Bone Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()
 
font = ('times', 16, 'bold')
title = Label(main, text='Identifying Bone Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Tumor X-Ray Images Dataset", command=uploadDataset)
uploadButton.place(x=300,y=605)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=725,y=605)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Trained CNN Bone Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=1220,y=605)
cnnButton.config(font=font1) 

svmButton = Button(main, text="Trained SVM Bone Tumor Detection Model", command=trainSVMDetectionModel)
svmButton.place(x=300, y=685)
svmButton.config(font=font1)

classifyButton = Button(main, text="Bone Tumor Segmentation & Classification", command=tumorClassification)
classifyButton.place(x=800,y=685)
classifyButton.config(font=font1)

graphButton = Button(main, text="CNN Training Accuracy Graph", command=graph)
graphButton.place(x=1220,y=685)
graphButton.config(font=font1)

compareButton = Button(main, text="Compare SVM and CNN", command=compareAlgorithms)
compareButton.place(x=600, y=750)
compareButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
