#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys  
import os 
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QShortcut
import viewer_gui
import numpy as np
import pandas as pd
import subprocess
import pickle
DATASET_PATH_1 = '/media/pavel/EC4D-D3EE/KutuzFullRealsenseRGBDataset'
DATASET_PATH_0 = '/media/pavel/EC4D-D3EE/SberKutuzDataset/Images'
def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return (int(resolution[0]), int(resolution[1]))


class ExampleApp(QtWidgets.QMainWindow, viewer_gui.Ui_MainWindow):
    def __init__(self):
        self.window_size = tuple(map(lambda x: int(x*0.9), get_screen_resolution()))
        self.pic_size = tuple(map(lambda x: int(x*0.8), self.window_size))
        super(ExampleApp, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.prev)  
        self.pushButton.clicked.connect(self.next)
        self.pushButton_3.clicked.connect(self.set_path)  

        self.curr_name = ""
        self.curr_pic = 0
        #self.setFixedSize(self.window_size[0], self.window_size[1])
        self.show()
        self.horizontalSlider.valueChanged.connect(self.slider_handler)
        self.label_9.setVisible(False)
        self.textEdit.setVisible(False)
        #self.pushButton_5.clicked.connect(self.next)
        self.shortcut_next = QShortcut(QKeySequence('Right'), self)
        self.shortcut_next.activated.connect(self.next)
        self.shortcut_prev = QShortcut(QKeySequence('Left'), self)
        self.shortcut_prev.activated.connect(self.prev)

        self.shortcut_hard = QShortcut(QKeySequence('h'), self)
        self.shortcut_hard.activated.connect(self.checkBox_12.click)

        self.shortcut_test = QShortcut(QKeySequence('t'), self)
        self.shortcut_test.activated.connect(self.checkBox_4.click)

        self.startup()        

    def set_path(self):

            self.label_8.setText("Go to: ")
            self.pushButton.setVisible(True)
            self.pushButton_2.setVisible(True)
            self.label.setVisible(True)
            
            self.label_7.setVisible(True)
            self.label_9.setVisible(True)
            self.lineEdit_2.setVisible(True)
            self.pushButton_4.setVisible(True)
            my_dir = QFileDialog.getExistingDirectory(self,"Open a folder","/media/pavel/EC4D-D3EE/SberKutuzDataset", QFileDialog.ShowDirsOnly)
            self.label_8.setText("Root dir is: ")
            self.lineEdit.setEnabled(False)
            self.DATASET_PATH = my_dir
            self.lineEdit.setText(my_dir)
            print(self.DATASET_PATH)


            with open(self.DATASET_PATH+'/files.pickle') as f:
                
                self.ind_list = pickle.load(f) #pd.read_csv(self.DATASET_PATH+"/radar_timestamps.csv")['global_ind'].tolist()
            try:
                self.hard_cases_list = np.array(pd.read_csv(self.DATASET_PATH+'/test_and_hard_cases.csv')['hc_flag'].tolist(), dtype=np.bool)
                self.test_cases_list = np.array(pd.read_csv(self.DATASET_PATH+'/test_and_hard_cases.csv')['test_flag'].tolist(), dtype=np.bool)
            except:
                self.hard_cases_list = np.zeros((len(self.ind_list)), dtype=np.bool)
                self.test_cases_list = np.zeros((len(self.ind_list)), dtype=np.bool)
            #self.ind_list = pd.read_csv(self.DATASET_PATH+"/radar_timestamps.csv")['global_ind'].tolist()
            print(self.ind_list)
            self.pushButton_4.clicked.connect(self.goto) 
            self.checkBox.clicked.connect(self.set_pics)  
            self.checkBox_2.clicked.connect(self.set_pics) 
            self.checkBox_11.clicked.connect(self.set_pics)
            self.checkBox_3.clicked.connect(self.set_pics)
            self.checkBox_12.clicked.connect(self.new_line)
            self.checkBox_4.clicked.connect(self.new_line)
            self.horizontalSlider.setRange(0, len(self.ind_list)-1)#self.ind_list[-1])
            self.horizontalSlider.setValue = self.ind_list[0]
            self.set_pics()

    def new_line(self):
        self.reset_hard_case()
        self.label_7.setText("Frame #{}. Curr batch is from {} to {}.     {} - Hard cases;  {} - Test cases".format(self.ind_list[self.curr_pic], self.ind_list[0], self.ind_list[-1], sum(self.hard_cases_list), sum(self.test_cases_list)))
       

    def startup(self):
            self.label_8.setText("Set root dir: ")
            self.pushButton.setVisible(False)
            self.pushButton_2.setVisible(False)
            self.label.setVisible(False)
            
            self.label_7.setVisible(False)
            self.label_9.setVisible(False)
            self.lineEdit_2.setVisible(False)
            self.pushButton_4.setVisible(False)

    def goto(self):
        if(self.lineEdit_2.text().isdigit()):
            g = int(self.lineEdit_2.text())
            if(g in self.ind_list):
                self.curr_pic = g-self.ind_list[0]
                self.set_pics()

    def set_pics(self):
        stack_list = []
        counter = 0
        self.image = cv2.resize(cv2.imread(self.DATASET_PATH+'/Images/'+str(self.ind_list[self.curr_pic])+".png"), (1280, 720))
        self.mask = cv2.resize(cv2.imread(self.DATASET_PATH+'/Semantic/'+str(self.ind_list[self.curr_pic])+".png"), (1280, 720))
        self.merged = cv2.addWeighted(self.image,0.99,self.mask,0.5,0)
        if(self.checkBox.isChecked()):   
            try:
                stack_list.append(cv2.resize(cv2.imread(self.DATASET_PATH+'/Images/'+str(self.ind_list[self.curr_pic])+".png"), (1280, 720)))
                #with open(self.DATASET_PATH+'/Images/DatasetGlassesNMirrors/FullZedRGBDataset/'+str(self.ind_list[self.curr_pic])+".png", 'rb') as f:
                #    stack_list.append(self.normalize(np.load(f), 'rs_depth'))
            except:
                stack_list.append(cv2.resize(cv2.imread('/home/pavel/Pictures/Screenshot from 2020-09-03 13-11-26.png'), (1280, 720)))
                print("Bad rs_depth #{}".format(self.ind_list[self.curr_pic]))
            stack_list[-1] = cv2.putText(stack_list[-1], "ZED_RGB", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            counter+=1
        if(self.checkBox_2.isChecked()):
            stack_list.append(cv2.imread(self.DATASET_PATH+'/Semantic/'+str(self.ind_list[self.curr_pic])+".png"))
            stack_list[-1] = cv2.putText(stack_list[-1], "Classes", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            counter+=1
        if(self.checkBox_11.isChecked()):
            stack_list.append(cv2.imread(self.DATASET_PATH+'/Objects/'+str(self.ind_list[self.curr_pic])+".png"))
            stack_list[-1] = cv2.putText(stack_list[-1], "Objects", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            counter+=1
        if(self.checkBox_3.isChecked()):
            stack_list.append(self.merged)
            stack_list[-1] = cv2.putText(stack_list[-1], "Merged", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            counter+=1
        
        
        diff = 6-counter
        if(counter == 1):
            pic = stack_list[0]
        elif(counter == 2):
            pic = np.hstack(stack_list)
        elif(counter == 3):
            upper = np.hstack([stack_list[0], stack_list[1]])
            down = np.hstack([stack_list[2], np.zeros(stack_list[2].shape, dtype = np.uint8)])
            pic = np.vstack([upper,down])
        elif(counter == 4):
            upper = np.hstack([stack_list[0], stack_list[1]])
            down = np.hstack([stack_list[2], stack_list[3]])
            pic = np.vstack([upper,down])
        
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        #pic = np.swapaxes(pic, 0,2)
        pic = cv2.resize(pic, self.pic_size)
        height, width, channel = pic.shape
        bytesPerLine = 3 * width
        qImg = QImage(pic.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(qImg))
        self.label.show()
        self.label_7.setText("Frame #{}. Curr batch is from {} to {}.     {} - Hard cases;  {} - Test cases".format(self.ind_list[self.curr_pic], self.ind_list[0], self.ind_list[-1], sum(self.hard_cases_list), sum(self.test_cases_list)))
       # self.horizontalSlider.value = 
        print(self.ind_list[self.curr_pic])
        self.checkBox_12.setChecked(self.hard_cases_list[self.curr_pic])
        self.checkBox_4.setChecked(self.test_cases_list[self.curr_pic])
        #self.horizontalSlider.setValue(self.ind_list[self.curr_pic])
        #cv2.imshow("FUCK", pic)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def normalize(self, pic, device):
        if(device == 'zed_depth'):
            #pic = pic*65535.0/20.0
            #pic = pic.astype(np.uint16) 
            pic = cv2.applyColorMap(cv2.convertScaleAbs(pic, alpha= 30), cv2.COLORMAP_JET)
        elif(device == 'rs_depth'):
            #pic = pic*65535.0/63703.0
            #pic = pic.astype(np.uint16)
            pic = pic/1000.0
            pic = cv2.applyColorMap(cv2.convertScaleAbs(pic, alpha= 30), cv2.COLORMAP_JET) 
        elif(device == 'rs_infra'):
            pic = pic*255/65535.0
            pic = pic.astype(np.uint8)
            pic = cv2.applyColorMap(cv2.convertScaleAbs(pic, alpha= 0.03), cv2.COLORMAP_JET) 
        elif(device == 'zed_conf'):
            
            pic = pic*255.0/110.0
            pic = pic.astype(np.uint8)
            pic = cv2.cvtColor(pic,cv2.COLOR_GRAY2RGB)
            print("conf shape ", pic.shape)
            #pic = cv2.applyColorMap(cv2.convertScaleAbs(pic, alpha= 3), cv2.COLORMAP_JET) 
        return pic

    def reset_hard_case(self):
        self.hard_cases_list[self.curr_pic] = self.checkBox_12.isChecked()
        self.test_cases_list[self.curr_pic] = self.checkBox_4.isChecked()
        temp = pd.DataFrame(columns=['frame', 'hc_flag', 'test_flag'])
        temp['frame'] = self.ind_list
        temp['hc_flag'] = self.hard_cases_list
        temp['test_flag'] = self.test_cases_list
        temp.to_csv(self.DATASET_PATH+'/test_and_hard_cases.csv')

    def prev(self):
        self.reset_hard_case()
        
        self.curr_pic -= 1
        if(self.curr_pic < 0):
            self.curr_pic = len(self.ind_list)-1
        self.set_pics()
        #self.horizontalSlider.setValue(self.ind_list[self.curr_pic])

    def next(self):
        self.reset_hard_case()
        self.curr_pic += 1
        if(self.curr_pic >= len(self.ind_list)):
            self.curr_pic = 0
        self.set_pics()
        #self.horizontalSlider.setValue(self.ind_list[self.curr_pic])

    def slider_handler(self, value):
        self.reset_hard_case()
        self.curr_pic = value#value-self.ind_list[0]
        self.set_pics()

def main():

    app = QtWidgets.QApplication(sys.argv) 
    window = ExampleApp() 
    window.show()  
    app.exec_()  

if __name__ == '__main__':  
    main() 
