import concurrent.futures
from kivy.config import Config
MIN_SIZE = (1000,650)
Config.set('graphics','width',MIN_SIZE[0])
Config.set('graphics','height',MIN_SIZE[1])
import kivy
from kivy.app import App
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import  ScreenManager,Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.properties import StringProperty
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.checkbox import CheckBox
from kivy.graphics.texture import Texture
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.scatter import Scatter
import time
import loadh5baend
import re
import cv2 as cv
import numpy as np
import os


class MainWindow(Screen):
    filePath = StringProperty('')
    original_img = StringProperty('')
    sizeX = 100
    sizeY = 100
    maintain_aspect = False
    def __init__(self,**kwargs):
        super(MainWindow,self).__init__(**kwargs)
        Window.bind(on_drop_file=self._on_file_drop)
        # Window.bind(on_drop_begin=self._on_drop_begin)
    def _on_file_drop(self,window,file_path,x,y):
        App.get_running_app().restart()
        # self.ids.vid.state = 'pause'
        # self.ids.vid.unload()
        self.filePath = file_path.decode("utf-8")
        self.original_img = self.filePath
        # if re.search('.mp4',self.filePath) != None:
        #     self.srcmode = 'video'
        #     self.ids.img.opacity = 0
        #     self.ids.vid.source = self.filePath
        #     self.ids.vid.state = 'play'
        #     self.ids.vid.options = {'eos': 'stop'}
        # else:
        self.srcmode = 'image'
        self.ids.vid.opacity = 0
        self.ids.img.source = self.filePath
        self.ids.img.reload()
    def on_enter_sizeX(self,text):
        sizeX = int(text)
        if sizeX <= 0:
            self.ids.warning0.opacity = 1
            self.ids.sizex.text = ''
        else:
            self.ids.warning0.opacity = 0
        if sizeX >= 2000:
            self.ids.warning1.opacity = 1
            self.ids.sizex.text = ''
        else:
            self.ids.warning1.opacity = 0
    def on_enter_sizeY(self,text):
        sizeY = int(text)
        if sizeY <= 0:
            self.ids.warning0.opacity = 1
            self.ids.sizey.text = ''
        else:
            self.ids.warning0.opacity = 0
        if sizeY >= 2000:
            self.ids.warning1.opacity = 1
        else:
            self.ids.warning1.opacity = 0
            self.ids.sizey.text = ''
    def on_checkAspect(self,value):
        if value:
            self.maintain_aspect = True
        else:
            self.maintain_aspect = False
    def on_resize(self):
        if self.ids.sizex.text != '' and self.ids.sizey.text != '':
            self.sizeX = int(self.ids.sizex.text)
            self.sizeY = int(self.ids.sizey.text)
        resize_img = resizer(self.original_img,dim=(self.sizeX,self.sizeY),maintain_aspect=self.maintain_aspect)
        cv.imwrite('resized_img.jpg',resize_img)
        self.filePath = 'resized_img.jpg'
        self.ids.img.source = self.filePath
        self.ids.img.reload()

    def on_return_original(self):
        self.filePath = self.original_img
        self.ids.img.source = self.filePath
        self.ids.img.reload()

    def reset(self,btn):
        App.get_running_app().restart()

    # def _on_drop_begin(self,window,x,y):
    #     self.ids.bgL.background_color = (0,1,1,0)
    #     self.ids.bgL.color = (1,0,0,1)
def resizer(ip_filepath,dim=(100,100),maintain_aspect=False):
    ip = cv.imread(ip_filepath)

    if maintain_aspect:
        size_max_ix = np.argmax(ip.shape)
        match(size_max_ix):
            case 1:
                if ip.shape[1] != dim[1] and np.max(ip.shape) == ip.shape[1]:
                    ratio_for_scaling = dim[1]/float(ip.shape[1])
                    dim = (dim[1],int(ip.shape[0]*ratio_for_scaling))
                    ip = cv.resize(ip,dim)
            case 0:
                if ip.shape[0] != dim[0] and np.max(ip.shape) == ip.shape[0]:
                    ratio_for_scaling = dim[0]/float(ip.shape[0])
                    dim = (int(ip.shape[1]*ratio_for_scaling),dim[0])
                    ip = cv.resize(ip,dim)
    else:
        ip = cv.resize(ip,dim)

    return ip
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

class h5DropDown(DropDown):
    pass
class AlgorithmDropDown(DropDown):
    pass
class DeNoiseDropDown(DropDown):
    pass
class ImageShow(Image,Widget):
    pass
class VideoShow(Video,Widget):
    pass
class ImageVideoWindow(Screen):
    def __init__(self,**kwargs):
        super(ImageVideoWindow,self).__init__(**kwargs)
        # h5_list = os.listdir("h5")[:n]
        # for url in h5_list:
        #     h5_urls.append(str('h5/'+url))

    btn = Button()
    h5 = None
    h5Select = 'full'
    alg = 'INTER_NEAREST'
    percentage = 1
    denoise = 'auto'
    denoiseAmount = 23
    def on_transit(self,Srcmode='empty',filePath=''):
        if Srcmode == 'image':
            org = ImageShow()
            self.ids['org'] = org
            org.size_bg = org.size
            org.pos_bg = org.pos
            org.background_color = .5,.5,.55,1
            org.source = filePath
            self.ids.showGrid.add_widget(org)
            superres = ImageShow()
            self.ids['superres'] = superres
            superres.size_bg = superres.size
            superres.pos_bg = superres.pos
            superres.background_color = .5,.5,.55,1
            self.ids.showGrid.add_widget(superres)
            Algorithm = ImageShow()
            self.ids['Algorithm'] = Algorithm
            Algorithm.size_bg = Algorithm.size
            Algorithm.pos_bg = Algorithm.pos
            Algorithm.background_color = .5,.5,.55,1
            self.ids.showGrid.add_widget(Algorithm)
        # if Srcmode == 'video':
        #     org = VideoShow()
        #     self.ids['org'] = org
        #     org.size_bg = org.size
        #     org.pos_bg = org.pos
        #     org.background_color = .5,.5,.55,1
        #     org.source = filePath
        #     self.ids.showGrid.add_widget(org)
        #     org.state = 'play'
        #     org.options = {'eos': 'stop'}
        #     svd = VideoShow()
        #     self.ids['svd'] = svd
        #     svd.size_bg = svd.size
        #     svd.pos_bg = svd.pos
        #     svd.background_color = .5,.5,.55,1
        #     self.ids.showGrid.add_widget(svd)
        #     svd.state = 'play'
        #     svd.options = {'eos': 'stop'}
    def on_denoise_dropdown(self,btn):
        deNoiseDropdown = DeNoiseDropDown()
        self.btn = btn
        btn.bind(on_release=deNoiseDropdown.open)
        deNoiseDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        deNoiseDropdown.bind(on_dismiss=self.get_User_choice_deNoise)
    def on_h5_dropdown(self, btn):
        hDropdown = h5DropDown()
        self.btn = btn
        btn.bind(on_release=hDropdown.open)
        hDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        hDropdown.bind(on_dismiss=self.get_User_choice_h5)
    def on_algorithm_dropdown(self,btn):
        aDropdown = AlgorithmDropDown()
        self.btn = btn
        btn.bind(on_release=aDropdown.open)
        aDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        aDropdown.bind(on_dismiss=self.get_User_choice_alg)
    def get_User_choice_h5(self,x):
        if self.btn.text != '' and self.btn.text!= 'Select h5':
            self.h5Select = self.btn.text
    def get_User_choice_alg(self,x):
        if self.btn.text != '' and self.btn.text != 'Select algorithm':
            self.alg = self.btn.text
    def get_User_choice_deNoise(self,x):
        if self.btn.text == 'Auto' or self.btn.text == 'None':
            self.ids.deNoisePercentInput.text = ''
            self.ids.deNoisePercentInput.disabled = True
        else:
            self.ids.deNoisePercentInput.disabled = False
        if self.btn.text != '' and self.btn.text != 'Select denoising':
            self.denoise = self.btn.text
    def on_process(self,btn,img):
        # if re.search('.mp4',img)!= None:
        #     if self.ids.rankinput.text != '':
        #         self.rankk = int(self.ids.rankinput.text)
        #     svdbaend.videosvd(img,rankk=self.rankk,rankOpt=self.rankOpt,mode = self.mode,multiThreaded = True)
        #     self.ids.svd.source = 'resultVid.mp4'
        #     self.ids.svd.reload()
        #     self.ids.svd.state = 'play'
        if self.ids.h5Input.text != '':
            self.h5 = int(self.ids.h5Input.text)
        if self.ids.deNoisePercentInput.text !='':
            self.denoiseAmount = int(self.ids.deNoisePercentInput.text)

        #Supperesolution pannel
        # svdbaend.imagesvd(img,rankk=self.rankk,rankOpt=self.rankOpt,mode = self.mode,forVid = False,multiThreaded = True)
        loadh5baend.predict(img,self.h5,self.denoiseAmount,self.denoise)
        self.ids.superres.source = 'DenoiseSuperres.jpg'
        suppres_info = cv.imread('DenoiseSuperres.jpg')
        # SNR = signaltonoise(suppres_info,axis=None)
        # SSIM = image.ssim_multiscale(tf.convert_to_tensor(suppres_info),tf.convert_to_tensor(suppres_info),np.max(np.reshape(suppres_info,[-1])))
        self.ids.outputSuperInfo.text = 'Size: '+str(int(suppres_info.shape[0]))+' '+str(int(suppres_info.shape[1]))+'\n'+'Kb: '+str(float(os.stat('DenoiseSuperres.jpg').st_size)/1024)+'\n'
        self.ids.superres.reload()


        #Algorithm pannel
        self.ids.algTitle.text = self.alg
        imgRead = cv.imread(img)
        alg_img = cv.resize(imgRead,(imgRead.shape[1]*4,imgRead.shape[0]*4),interpolation=getattr(cv,self.alg))
        cv.imwrite('alg_img.jpg',alg_img)
        alg_img_info = cv.imread('alg_img.jpg')
        # SNR = signaltonoise(alg_img_info,axis=None)
        self.ids.outputAlgInfo.text = 'Size: '+str(int(alg_img_info.shape[0]))+' '+str(int(alg_img_info.shape[1]))+'\n'+'Kb: '+str(float(os.stat('alg_img.jpg').st_size)/1024)+'\n'
        self.ids.Algorithm.source = 'alg_img.jpg'
        self.ids.Algorithm.reload()
    def on_enter_h5(self,text):
        num = int(text)
        if num <= 0:
            self.ids.warning.opacity = 1
            self.ids.h5Input.text = ''
        else:
            self.ids.warning.opacity = 0
    def on_enter_percentDenoise(self,text):
        num = int(text)
        if num<=0:
            self.ids.warningPercentange.text = 'Must be larger than zero!'
            self.ids.warningPercentange.opacity = 1
            self.ids.deNoisePercentInput.text = ''
        elif num>100:
            self.ids.warningPercentange.text = 'Must be smaller than 100%!'
            self.ids.warningPercentange.opacity = 1
            self.ids.deNoisePercentInput.text = ''
        else:
            self.ids.warningPercentange.text = ''
            self.ids.warning.opacity = 0
class WebcamWindow(Screen):
    def __init__(self, **kwargs):
        super(WebcamWindow, self).__init__(**kwargs)

        # org = Image()
        # self.ids['org'] = org
        # self.ids.showGrid.add_widget(org)
        #
        # svd = Image()
        # self.ids['svd'] = svd
        # self.ids.showGrid.add_widget(svd)

    btn = Button()
    rankk = None
    rankOpt = 'full'
    mode = 'rgb'

    def on_transit(self):
        self.capture = cv.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 64.0)
        org = Image()
        self.ids['org'] = org
        self.ids.showGrid.add_widget(org)

        svd = Image()
        self.ids['svd'] = svd
        self.ids.showGrid.add_widget(svd)
    def update(self, dt):
        ret, frame = self.capture.read()
        frame = cv.resize(frame, dsize=None, fx=0.6, fy=0.6)
        svdframe = np.array(svdbaend.imagesvd(frame,rankk = self.rankk,rankOpt = self.rankOpt,mode = self.mode))
        # convert it to texture
        buf1 = cv.flip(frame, 0)
        buf11 = buf1.tostring()

        buf2 = cv.flip(svdframe, 0)
        buf22 = buf2.tostring()
        colorfmt = 'bgr'
        if self.mode == 'gs':
            colorfmt = 'luminance'
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture2 = Texture.create(size=(svdframe.shape[1], svdframe.shape[0]), colorfmt=colorfmt)
        texture1.blit_buffer(buf11,colorfmt='bgr', bufferfmt='ubyte')
        texture2.blit_buffer(buf22,colorfmt=colorfmt, bufferfmt='ubyte')

        # display image from the texture
        self.ids.org.texture = texture1
        self.ids.svd.texture = texture2
    def on_reload(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.ids.showGrid.clear_widgets()
        self.on_transit()
        if self.ids.rankinput.text != '':
            self.rankk = int(self.ids.rankinput.text)
    def on_back(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.btn = Button()
        self.rankk = None
        self.rankOpt = 'full'
        self.mode = 'rgb'
    def on_rank_dropdown(self, btn):
        rDropdown = RankDropDown()
        self.btn = btn
        btn.bind(on_release=rDropdown.open)
        rDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        rDropdown.bind(on_dismiss=self.get_User_choice_rank)
    def on_mode_dropdown(self,btn):
        mDropdown = ModeDropDown()
        self.btn = btn
        btn.bind(on_release=mDropdown.open)
        mDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        mDropdown.bind(on_dismiss=self.get_User_choice_mode)
    def get_User_choice_rank(self,x):
        if self.btn.text != '' and self.btn.text!= 'Select rank options':
            self.rankOpt = self.btn.text
    def get_User_choice_mode(self,x):
        if self.btn.text != '' and self.btn.text != 'Select mode':
            self.mode = self.btn.text
    def on_enter_rank(self,text):
        num = int(text)
        if num <= 0:
            self.ids.warning.opacity = 1
        else:
            self.ids.warning.opacity = 0

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("srgantemplate.kv")

class MyMainApp(App):
    def check_resize(self,instance,x,y):
        if x < MIN_SIZE[0]:
            Window.size = (1000 , Window.size[1])
        if y < MIN_SIZE[1]:
            Window.size = (Window.size[0],650)
    def restart(self):
        Imageshow = self.root.screens[0].children[2]
        Imageshow.source = str(None)
        Videoshow = self.root.screens[0].children[1]
        Videoshow.state = 'pause'
        Videoshow.unload()
        Videoshow.source = str(None)
        self.root.screens[0].srcmode = 'empty'
        self.root.screens[1].children[0].children[1].srcmode = 'empty'
        self.root.screens[1].children[0].children[1].clear_widgets()
        self.root.screens[0].children[0].children[1].opacity = 1
    def build(self):
        Window.bind(on_resize=self.check_resize)
        return kv
if __name__=="__main__":
    MyMainApp().run()


