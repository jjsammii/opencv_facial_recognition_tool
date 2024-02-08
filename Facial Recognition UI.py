from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import pickle
import sys
import os
from PIL import Image, ImageTk


root = Tk()
root.title("Guardsman Ltd Facial recognition GUI")
root.minsize(width=300, height=300)
root.configure(background="white")
photo = PhotoImage(file="photo2.gif")
Label(root, image=photo, bg="white").grid(row=0,column=0, columnspan=4, rowspan=1, sticky=W+E+N+S, padx=5, pady=5)
menu = Menu(root)
root.config(menu=menu)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
settings = {'scaleFactor': 1.3, 'minNeighbors': 3, 'minSize': (50, 50), 'flags': cv2.CASCADE_SCALE_IMAGE}

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(1)

def doNothing():
	print("print nothing")

def normalize_intensity(images):
    # Store all grayscale range intensities
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(100, 100)):

    #Store all images with the same pixel size
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # using different OpenCV method if enlarging or shrinking
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm
           

def scanFace():    
    while(True):        
        #Capture frame by frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')        
        for(x, y, w, h) in faces:
            #print(x, y, w, h)
            roi_g = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = (roi_g)

        # recognize faces: use deep learned model predict keras tensorflow pytorch scikit learn
        #def scanFace():
            
            output.delete(0.0, END)   
            id_, conf = recognizer.predict(roi_gray)        
            if conf<60:
                print(conf)
                print("Unknown face, Access Denied")
                definition = "Person Being Scanned: Unknown "    
            else:
                print(id_)
                print(conf)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                definition = ("Person being scanned:"+ " " + name)
            output.insert(END, definition)           
                
            #img_item = "my-image.png"
            #cv2.imwrite(img_item, roi_gray)

            color = (255, 0, 0) #BGR 0 - 255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        
        #Diplay the resulting frame
        cv2.imshow('GML Facial Recognition in Progress',frame)           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#scanFace()

def CropSFace():
    InputName = textentry.get()
    output.delete(0.0, END)          
    folder = "Images/" + str(textentry).lower()# input name
    final_path = os.path.join('Images/'+ str(textentry.get()))
    if not os.path.exists(final_path):        
        os.mkdir(final_path) 
        count = 0       
        timer = 0
        definition = "Upload Completed"        
    else:
        definition = "This Name already Exists/ No input Provided"
    output.insert(END, definition)

    while True and count < 10:

    # Capture frame-by-frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30))        
    

            # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        crop_face = []
        if len (faces) and timer % 700 == 50: #every Second or so
            for (x, y, w, h) in faces:
                crop_face.append(gray[y: y + h, x: x + w])
                cropface1 = normalize_intensity(crop_face)
                cutface = resize (cropface1)                           
                cv2.imwrite(final_path + '/' + str(count) + '.jpg', cutface[0])
                count += 1      
    
        
    # Display the resulting frame
        cv2.imshow('Guardsman Employess Face Scan', frame)
        #cv2.imwrite(folder + '/' + str(crop_face) + '.jpg', faces)     
    
    
        cv2.waitKey(1) & 0xFF == ord('q')
        timer += 50
            #break

 #When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def ConvertFace():

    current_id = 0
    label_ids ={}
    y_labels = []
    x_train = []
    import pickle



    output.delete(0.0, END)

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "_").lower()
                #print(label, path)
                if not label in label_ids:
                    label_ids[label]= current_id
                    current_id += 1
                id_ = label_ids[label]
                #print(label_ids)
                #y_labels.append(label)# some number
                #x_train.append(path) # verify this image, turn into a NUMPY array, GRAY
                pil_image = Image.open(path).convert("L") # grayscale
                size = (550, 550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_image, "uint8")
                #print(image_array)
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

                for(x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    print(y_labels)
    print(x_train)


    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    definition =  "Training Complete"
    
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")
    output.insert(END, definition)
#ConvertFace()




def refresh_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action"""
    root.destroy()
    python = sys.executable
    os.execl(python, python, * sys.argv)
  
def OpenFile():
    name = askopenfilename(initialdir="Images/", filetypes =(("Image File", "*.jpg"),("All Files","*.*")), title = "Choose a file.")
    print (name)
    #Using try in case user types in unknown file or closes without choosing a file.
    try:
        with open(name,'r') as UseFile:
            print(UseFile.read())
    except:
        print("No file exists")

subMenu = Menu(menu)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Refresh...", command=refresh_program)
subMenu.add_command(label="Manaage Employees", command=OpenFile)
subMenu.add_separator()
subMenu.add_command(label="Exit", command=doNothing)

editMenu = Menu(menu)
menu.add_cascade(label="Edit", menu=editMenu)
editMenu.add_command(label="Redo", command=doNothing)

# ****Toolbar****

Button(root, text="Activate Live Stream", width = 20, command=scanFace).grid(row=1, column=0, columnspan=1, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)

Button(root, text="Train AI Database", command=ConvertFace).grid(row=1, column=1, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)

Button(root, text="Manage Employees", command=OpenFile).grid(row=1, column=3, columnspan=1, rowspan=1, sticky=W+E+N+S, padx=5, pady=5)
Button(root, text="Log Out", command=root.destroy).grid(row=1, column=4, columnspan=1, rowspan=1, sticky=W+E+N+S, padx=5, pady=5)
#quitbutton.pack(side=LEFT, padx=2, pady=2)

Label(root, text="Enter Employee Name", bg="light blue", fg="white", font="none 12 bold").grid(row=7, column = 0, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)

textentry = Entry(root, width=20, bg="light grey")
textentry.grid(row=9, column=0, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)
#textentry.pack(side=CENTER, padx=2, pady=2)
Button(root, text="Scan & Upload Employee Face", command=CropSFace).grid(row=11, column=0, sticky=W)
Button(root, text="Refresh", command=refresh_program).grid(row=11, column=1, sticky=W)
Label(root, text="Perform AI Training and then refresh after employee facial upload", fg="black", font="none 10 bold").grid(row=11, column = 2, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)
Label(root, text="\n Status Report:", bg="light blue", fg="white", font="none 12 bold").grid(row=13, column = 0, columnspan=2, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)
output = Text(root, width=20, height=2, fg = "red", bg="light grey", font="none 10 bold")
output.grid(row=15,column=0, columnspan=2, rowspan=1, sticky=W+E+N+S, padx=5, pady=5)

# **** Status Bar *****
status = Label(root, text="Press 'q' to exit live stream", bd=1, relief=SUNKEN, anchor=W)
status.grid(row=26, column=0, columnspan=4, rowspan=4, sticky=W+E+N+S, padx=5, pady=5)

root.mainloop()

