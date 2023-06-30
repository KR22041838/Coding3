
# Sign Language App Project
There are three Coding sections to this Project:
- Pose.Net: Final Pose.Net Sign
- Gesture Recognition: Final Sign Language Notebook
- Neural Network: Final Training NN

For this project I wanted to use sign language to create a model which translated images of British sign language finger spelling into letters. With the idea of bringing sign language into primary school education to help with learning to spell as studies suggest children learn faster when combining the two and it can aid in things such as dyslexia. The end goal would be to create a game app where children our given verbal prompts or images such as animals and a camera reads the child’s finger spelling in real time and prints it onto the screen where they can see if they have the correct spelling. The app could be used in class or to aid spelling practice outside of school on their own. 

I started off the project by looking at real-time pose classifiers that I could train to read signs. I chose to experiment with a the pre-made pose classifier model which uses interactive machine learning called pose.net  by louis-dev. As the model classifies poses rather than hand gestures I knew it will have limits in its purpose for this project. However I wanted to experiment with how accurate I could get the classifications and how an app interface might look for the project, plus I was struggling to find any easy-to-use off-the-shelf gesture recognition models. 

I started off training the model on AEIOU which corresponds to pointing to the fingers on the right hand. The model was reasonably accurate when identifying A, E and I but OU became less accurate as the arm positions are very similar. Then I tried a ABCDE as these gestures were more varied in appearance and I ensured the position of my arms were different for each one, the model became more accurate at identifying the letter being signed when using my arms in this way however as the signs become more complicated the variation in arm position becomes less and less. 

# The changes I made to the Pose.Net code:
Instead of having only one letter showing on the screen at a time I wanted the letters to stay on the screen to be able to create words so I added  ‘textElement.textContent += " " + text;’. However I now the letters were printing too fast across the screen and I was not able to create words. I tried to delay the printing of the letters but this only created lag and did not fix the issue. Instead added I block of code that included an ‘inputCountThreshold = 5’ to slow down the sign recognition rate, so now it waits for 5 inputs to pass before giving an output letter. 

I wanted to add an extra layer of accuracy that I was struggling to achieve in the training data itself, so I added a block of code which takes the average of the 5 poses dictated by the inputCountThreshold and produces the average output pose which made the model more accurate. 

It was at this point however I had fully realised the limitations of using Pose.Net for the task of recognising sign language, I had learnt what I wanted to learn in terms of making and fine tuning the accuracy of a data and how I might want the user interface to interact with the user if it was made into an App. But it was time to move on to and actual gesture recognising model.

# Gesture Recognizer
The model I decided to experiment further with was the MediaPipe Gesture Recognizer. It can identify hand gestures in real time, video or image and provides landmarks for specific points on the hand that were detected.

The code example MediaPipe provided, runs the task of analysing the photo input, and outputting 21 landmarks per hand as well as signposting left and right hands. This allows you to create your own data to feed into and train a your own neural network. 

The notebook ran well in google colab but I prefer to use Jupyter notebook so after setting up the python environment I transferred the code into Jupyter notebook. The notebook did not work initially and I was having trouble transferring in and reading the google colab files. Looking back I should have stuck with colab but I spent a lot of time just trying to get the Jupyter to work. 

After failing to access the google colab photo set, I removed that part of the code and added a block of code using Chat GTP to load my own photos instead which worked well.

The RGB values of the photos are slightly off, I thought they may be inverted but when I inverted them it became clear that was not the case. I tried other colour values but nothing worked, as it does not affect the final outcome I decided to leave it. 

The original google colab output placed a percentage accuracy next to each photo but this got lost somewhere in the change over to jupyter. 

In Jupyter I could only see faint lines on the output photos as the printed landmarks and adjoining lines were too small. I added code to increase line thickness and circle radius and I am now happy with the working of the notebook and the output. 


Next up I needed to make a data set. I took my own photos of BSL signs ‘a’ and ‘b’ and ran them through the notebook and added code so it prints the landmark data. It was at this point I realised there with 42 x, y and z coordinates to record per landmark meaning I had to input 126 data values per sign photo. Because of time restraints I only managed to record 7 photos worth of data and split the data 5 to 2 for training and testing in cvs files. 

To train a Neural Network on this data I revisited a notebook from week 1. I needed to make changes by loading in my own data set the ‘input_columns =’ for the 42 x,y,z values. There was an issue with how the signs were named for the code to read so I mapped them to numbers instead e.g ‘A_1’: 1,  ‘A_2’: 2 etc. 
I had a lot of problems with how he code read the cvs.files, and kept getting the error ‘Data cardinality is ambiguous’ followed by x and y number which were not correct. At first I thought I had made my cvs files incorrectly but after researching into it and creating debug cvs  files with only one set of x,y,z coordinates it still came up with the error. Next I tried using pandas to write my own data inside the notebook and save it to its own cvs.file just encase it was the programs I was using. I still came up with this issue, and after using the cvs files in other notebooks and found they worked/read as intended I came to the conclusion It must be something in the code that is causing the conflict. Unfortunately at this point I had ran out of time. 

If my code did work I had intended to use the CNN to train the model on recognising signs based off of the landmarks, so I can add a new photo in and it classifies the sign. I would also need to fine tune the parameters to increased the accuracy. The next step would have been to combine the Gesture Recognizer notebook with the training model to see if I can import the 126 landmark values per photo directly into the model to classify rather than recording it by hand like the current cvs. Files. After this I would try to incorporate live stream gesture recognition to steer the project in the direction of making an Sign Language App.

Links for Code and Tutorial:
Gesture Recogniser:
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#scrollTo=KHqaswD6M8iO
Gesture Recogniser Documentation:
https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python 
Pose.Net Mimic Code:
https://mimicproject.com/code/48d5b6d9-794e-97d4-a16e-4780cf6c4a8c
Neural Network Notebook:
https://git.arts.ac.uk/rfiebrink/ExploringMachineIntelligence_Spring2023/tree/main/week1
W3schools:
https://www.w3schools.com/jsref/prop_node_textcontent.asp
Chat GTP.

 
