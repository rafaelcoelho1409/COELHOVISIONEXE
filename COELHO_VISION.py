from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QPushButton, 
    QWidget, 
    QComboBox, 
    QTabWidget, 
    QSpacerItem,
    QSizePolicy,
    QFrame,
    QFileDialog,
    QMessageBox
)
from PyQt6.QtGui import (
    QFont, 
    QPixmap, 
    QImage,
    QDesktopServices,
    QIcon)
from PyQt6.QtCore import (
    Qt, 
    QTimer,
    QUrl,
    QSize)
import cv2
import pyautogui
import logging
import os
import datetime as dt
from functions import (
    resource_path,
    NoModel,
    FullFaceDetector,
    MediaPipeObjectDetection,
    MediaPipeFaceDetector,
    MediaPipeImageSegmentation,
    MediaPipeHandLandmarker,
    MediaPipePoseEstimation,
    RFObjectDetection,
    RFTracking,
    OpenVINOImageSegmentation,
    OpenVINODepthEstimation,
    OpenVINOPoseEstimation,
    OpenVINOOCR
)

def map_all_cameras():
    available_cameras = []
    for i in range(20):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)  # Attempt to open the camera
            if cap.isOpened():  # If the camera opened successfully
                available_cameras.append(str(i))
                cap.release()  # Release the camera
            else:
                cap.release()
        except Exception as e:
            logging.exception(f"An error occurred while checking camera index {i}.", e)
            continue
    return available_cameras

app = QApplication([])
app.setStyle("fusion")

FULL_SCREEN_MODE = False
RES_WIDTH, RES_HEIGHT = 640, 480
MAP_ALL_CAMERAS = map_all_cameras()
try:
    AVAILABLE_CAMERA = MAP_ALL_CAMERAS[0]
except Exception as e:
    # If an exception occurs, show a critical QMessageBox
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText("There are no available cameras. Make sure you have at least one camera connected to your device.")
    msg.setWindowTitle("Available camera error")
    msg.exec()
CURRENT_RESOLUTION = "640 x 480"
SAVE_COUNT = 1
CAP = cv2.VideoCapture(int(AVAILABLE_CAMERA))

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = NoModel()
        self.font_scale = min(
            int((pyautogui.size()[0] * 20) / 1920),
            int((pyautogui.size()[1] * 20) / 1080),
        )
        #self.showFullScreen()
        self.showMaximized()
        self.setWindowIcon(QIcon(resource_path(os.path.join('assets', 'coelho_vision_icon.png'))))
        self.initUI()
        self.mainLayout()

    def initUI(self):
        # Set the main window properties
        self.setWindowTitle("COELHO VISION")
        # Create the main layout
        self.main_layout = QVBoxLayout()
        # Title and logo horizontal layout
        title_logo_layout = QHBoxLayout()
        sub_layout = QHBoxLayout()
        #TITLE LOGO LAYOUT
        # Title label
        title = QLabel("COELHO VISION")
        title.setFont(
            QFont("Times New Roman", 35, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignCenter)
        # Logo label
        logo = QLabel()
        pixmap = QPixmap(resource_path(os.path.join('assets', 'coelho_vision_logo.png')))  # Replace with the path to your logo image
        logo.setPixmap(
            pixmap.scaled(
                int(pixmap.width() * 0.075), 
                int(pixmap.height() * 0.075), 
                Qt.AspectRatioMode.KeepAspectRatio))  # Adjust scaling as needed
        logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        # Spacer to push the title to the left and the logo to the right
        spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # Add the title, spacer, and logo to the title logo layout
        author = QLabel("Author: Rafael Silva Coelho")    
        author.setAlignment(Qt.AlignmentFlag.AlignRight) 
        title_logo_layout.addWidget(title)
        title_logo_layout.addItem(spacer)
        title_logo_layout.addWidget(logo)
        sub_layout.addWidget(author)
        spacer2 = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        web_version_button = QPushButton("Web version")
        web_version_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://coelhovision.streamlit.app/"))
        )
        web_version_button.setFixedWidth(int(pixmap.width() * 0.075))
        sub_layout.addWidget(author)
        sub_layout.addItem(spacer2)
        sub_layout.addWidget(web_version_button)
        self.main_layout.addLayout(title_logo_layout)
        self.main_layout.addLayout(sub_layout)
        author = QLabel("Author: Rafael Silva Coelho")
        author.setAlignment(Qt.AlignmentFlag.AlignRight)
        # Set the main layout to the central widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def mainLayout(self):
        # Add tabs
        tabs = QTabWidget()
        #----------------------------------------------------------------------
        self.home_tab = self.HomeTab()
        self.object_detection_page = self.ObjectDetectionPage()
        self.image_segmentation_page = self.ImageSegmentationPage()
        self.pose_estimation_page = self.PoseEstimationPage()
        self.about_us_page = self.AboutUsPage()
        #----------------------------------------------------------------------
        tabs.addTab(self.home_tab, "Home")
        tabs.addTab(self.object_detection_page, "Object Detection")
        tabs.addTab(self.image_segmentation_page, "Image Segmentation")
        tabs.addTab(self.pose_estimation_page, "Pose Estimation")
        tabs.addTab(self.about_us_page, "About Us")
        # Add more tabs as necessary
        self.main_layout.addWidget(tabs)
        #main filters
        filters_content = QWidget()
        filters_layout = QHBoxLayout()
        self.available_cameras_legend = QLabel("Available cameras")
        self.available_cameras_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resolution_legend = QLabel("Frame resolution")
        self.resolution_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.available_cameras_box = QComboBox()
        self.available_cameras_box.addItems(MAP_ALL_CAMERAS)
        self.available_cameras_box.currentTextChanged.connect(self._available_cameras)
        #self.available_cameras_box.setCurrentIndex(self.available_cameras_box.findText(str(AVAILABLE_CAMERA)))
        self.actual_resolution = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        self.resolution_box = QComboBox()
        self.resolution_box.addItems([
            "640 x 480",
            "800 x 600",
            "1024 x 768",
            "1280 x 720",
            "1920 x 1080"
        ])
        self.resolution_box.currentTextChanged.connect(self._set_resolution)
        self.resolution_box.setCurrentIndex(self.resolution_box.findText(CURRENT_RESOLUTION))
        self.full_screen_btn = QPushButton("Full Screen")
        #self.full_screen_btn.setFixedHeight(int(50))
        #self.full_screen_btn.setCheckable(True)
        self.full_screen_btn.clicked.connect(self._open_full_screen)
        self.save_image_btn = QPushButton("Save Image")
        #self.save_image_btn.setFixedHeight(int(50))
        self.save_image_btn.clicked.connect(self._save_image)
        for x in [
            self.available_cameras_legend, 
            self.available_cameras_box, 
            self.resolution_legend,
            self.resolution_box,
            self.save_image_btn,
            self.full_screen_btn
        ]:
            filters_layout.addWidget(x)
        filters_content.setLayout(filters_layout)
        self.main_layout.addWidget(filters_content)
        # Set main layout to a central widget since QMainWindow requires it
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def HomeTab(self):
        image_scale = min(
            (pyautogui.size()[0] * 0.5) / 1920,
            (pyautogui.size()[1] * 0.5) / 1080,
        )
        content = QWidget()
        layout = QVBoxLayout()
        font_italic = QFont("Sans Serif", 20)
        font_italic.setItalic(True)
        #OBJECT DETECTION
        od = QLabel("Object Detection")
        od.setFont(font_italic)
        od.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        od_images_widget = QWidget()
        od_images = QHBoxLayout()
        img_fullfacedection = QLabel()
        img_fullfacedection_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_fullfacedetector.png')))
        img_fullfacedection.setPixmap(img_fullfacedection_pixmap.scaled(
                int(img_fullfacedection_pixmap.width() * image_scale), 
                int(img_fullfacedection_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_facedection = QLabel()
        img_facedection_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_facedetection.png')))
        img_facedection.setPixmap(img_facedection_pixmap.scaled(
                int(img_facedection_pixmap.width() * image_scale), 
                int(img_facedection_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_objectdetection = QLabel()
        img_objectdetection_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_objectdetection.png')))
        img_objectdetection.setPixmap(img_objectdetection_pixmap.scaled(
                int(img_objectdetection_pixmap.width() * image_scale), 
                int(img_objectdetection_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_objecttracking = QLabel()
        img_objecttracking_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_objecttracking.jpg')))
        img_objecttracking.setPixmap(img_objecttracking_pixmap.scaled(
                int(img_objecttracking_pixmap.width() * image_scale), 
                int(img_objecttracking_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_ocr = QLabel()
        img_ocr_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_ocr.png')))
        img_ocr.setPixmap(img_ocr_pixmap.scaled(
                int(img_ocr_pixmap.width() * image_scale), 
                int(img_ocr_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        for x in [
            img_fullfacedection, 
            img_facedection, 
            img_objectdetection,
            img_objecttracking,
            img_ocr]:
            od_images.addWidget(x)
        od_images_widget.setLayout(od_images)
        od_captions_widget = QWidget()
        od_captions = QHBoxLayout()
        caption_fullfacedetector = QLabel("Full Face Detector")
        caption_facedetection = QLabel("Face Detection")
        caption_objectdetection = QLabel("Object Detection")
        caption_objecttracking = QLabel("Object Tracking")
        caption_ocr = QLabel("Optical Character Recognition")
        for x in [
            caption_fullfacedetector,
            caption_facedetection, 
            caption_objectdetection,
            caption_objecttracking,
            caption_ocr]:
            od_captions.addWidget(x)
        od_captions_widget.setLayout(od_captions)
        #IMAGE SEGMENTATION
        iseg = QLabel("Image Segmentation")
        iseg.setFont(font_italic)
        iseg.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        iseg_images_widget = QWidget()
        iseg_images = QHBoxLayout()
        img_imagesegmentation = QLabel()
        img_imagesegmentation_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_imagesegmentation.png')))
        img_imagesegmentation.setPixmap(img_imagesegmentation_pixmap.scaled(
                int(img_facedection_pixmap.width() * image_scale), 
                int(img_facedection_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_depthestimation = QLabel()
        img_depthestimation_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_depthestimation.png')))
        img_depthestimation.setPixmap(img_depthestimation_pixmap.scaled(
                int(img_depthestimation_pixmap.width() * image_scale), 
                int(img_depthestimation_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_semanticsegmentation = QLabel()
        img_semanticsegmentation_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_semanticsegmentation.png')))
        img_semanticsegmentation.setPixmap(img_semanticsegmentation_pixmap.scaled(
                int(img_semanticsegmentation_pixmap.width() * image_scale), 
                int(img_semanticsegmentation_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        for x in [
            img_imagesegmentation,
            img_depthestimation,
            img_semanticsegmentation]:
            iseg_images.addWidget(x)
        iseg_images_widget.setLayout(iseg_images)
        iseg_captions_widget = QWidget()
        iseg_captions = QHBoxLayout()
        caption_imagesegmentation = QLabel("Image Segmentation")
        caption_depthestimation = QLabel("Depth Estimation")
        caption_semanticsegmentation = QLabel("Semantic Segmentation")
        for x in [
            caption_imagesegmentation,
            caption_depthestimation,
            caption_semanticsegmentation]:
            iseg_captions.addWidget(x)
        iseg_captions_widget.setLayout(iseg_captions)
        #POSE ESTIMATION
        pe = QLabel("Pose Estimation")
        pe.setFont(font_italic)
        pe.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        pe_images_widget = QWidget()
        pe_images = QHBoxLayout()
        img_handlandmarker = QLabel()
        img_handlandmarker_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_handlandmarker.png')))
        img_handlandmarker.setPixmap(img_handlandmarker_pixmap.scaled(
                int(img_handlandmarker_pixmap.width() * image_scale), 
                int(img_handlandmarker_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        img_poseestimation = QLabel()
        img_poseestimation_pixmap = QPixmap(resource_path(os.path.join('assets', 'home_poseestimation.png')))
        img_poseestimation.setPixmap(img_poseestimation_pixmap.scaled(
                int(img_poseestimation_pixmap.width() * image_scale), 
                int(img_poseestimation_pixmap.height() * image_scale), 
                Qt.AspectRatioMode.KeepAspectRatio))
        for x in [
            img_handlandmarker, 
            img_poseestimation]:
            pe_images.addWidget(x)
        pe_images_widget.setLayout(pe_images)
        pe_captions_widget = QWidget()
        pe_captions = QHBoxLayout()
        caption_handlandmarker = QLabel("Hand Landmarker")
        caption_poseestimation = QLabel("Pose Estimation")
        for x in [
            caption_handlandmarker, 
            caption_poseestimation]:
            pe_captions.addWidget(x)
        pe_captions_widget.setLayout(pe_captions)
        #------------------------------------------------------------------------------
        layout.addWidget(od)
        layout.addWidget(od_images_widget)
        layout.addWidget(od_captions_widget)
        layout.addWidget(self._divider())
        layout.addWidget(iseg)
        layout.addWidget(iseg_images_widget)
        layout.addWidget(iseg_captions_widget)
        layout.addWidget(self._divider())
        layout.addWidget(pe)
        layout.addWidget(pe_images_widget)
        layout.addWidget(pe_captions_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.setLayout(layout)
        return content
    
    def ObjectDetectionPage(self):
        content = QWidget()
        layout = QVBoxLayout()
        subtitle = QLabel("Object Detection")
        subtitle.setFont(QFont("Times New Roman", self.font_scale, QFont.Weight.Bold))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        #FILTERS
        filters_legend_content = QWidget()
        filters_legend_layout = QHBoxLayout()
        filters_legend_layout.addWidget(QLabel("Role"))
        filters_legend_content.setLayout(filters_legend_layout)
        filters_box_content = QWidget()
        filters_box_layout = QHBoxLayout()
        role_box1 = QComboBox()
        role_box1.addItems([
            "-",
            "Full Face Detection",
            "Face Detector (MediaPipe)",
            "Object Detection (MediaPipe)",
            "Object Detection (RoboFlow)",
            "Object Tracking (RoboFlow)",
            "Optical Character Recognition (OpenVINO)"
        ])
        role_box1.currentTextChanged.connect(self._role)
        filters_box_layout.addWidget(role_box1)
        filters_box_content.setLayout(filters_box_layout)
        #TIMER
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(0)  # Adjust the interval to match the frame rate you desire
        self.vision_content = QWidget()
        self.vision_layout = QVBoxLayout()
        self.vision_label1 = QLabel()
        self.vision_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vision_layout.addWidget(self.vision_label1)
        self.vision_content.setLayout(self.vision_layout)
        layout.addWidget(subtitle)
        layout.addWidget(filters_legend_content)
        layout.addWidget(filters_box_content)
        layout.addWidget(self.vision_label1)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.setLayout(layout)
        return content
    
    def ImageSegmentationPage(self):
        content = QWidget()
        layout = QVBoxLayout()    
        subtitle = QLabel("Image Segmentation")
        subtitle.setFont(QFont("Times New Roman", self.font_scale, QFont.Weight.Bold))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        #FILTERS
        filters_legend_content = QWidget()
        filters_legend_layout = QHBoxLayout()
        filters_legend_layout.addWidget(QLabel("Role"))
        filters_legend_content.setLayout(filters_legend_layout)
        filters_box_content = QWidget()
        filters_box_layout = QHBoxLayout()
        role_box2 = QComboBox()
        role_box2.addItems([
            "-",
            "Image Segmentation (MediaPipe)",
            "Depth Estimation (OpenVINO)",
            "Semantic Segmentation - Mask Mode (OpenVINO)",
            "Semantic Segmentation (OpenVINO)"
        ])
        role_box2.currentTextChanged.connect(self._role)
        filters_box_layout.addWidget(role_box2)
        filters_box_content.setLayout(filters_box_layout)
        self.vision_content = QWidget()
        self.vision_layout = QVBoxLayout()
        self.vision_label2 = QLabel()
        self.vision_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vision_layout.addWidget(self.vision_label2)
        self.vision_content.setLayout(self.vision_layout)
        layout.addWidget(subtitle)
        layout.addWidget(filters_legend_content)
        layout.addWidget(filters_box_content)
        layout.addWidget(self.vision_label2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.setLayout(layout)
        return content
    
    def PoseEstimationPage(self):
        content = QWidget()
        layout = QVBoxLayout()    
        subtitle = QLabel("Pose Estimation")
        subtitle.setFont(QFont("Times New Roman", self.font_scale, QFont.Weight.Bold))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        #FILTERS
        filters_legend_content = QWidget()
        filters_legend_layout = QHBoxLayout()
        filters_legend_layout.addWidget(QLabel("Role"))
        filters_legend_content.setLayout(filters_legend_layout)
        filters_box_content = QWidget()
        filters_box_layout = QHBoxLayout()
        role_box3 = QComboBox()
        role_box3.addItems([
            "-",
            "Hand Landmarker (MediaPipe)",
            "Pose Estimation (MediaPipe)",
            "Pose Estimation (OpenVINO)",
        ])
        role_box3.currentTextChanged.connect(self._role)
        filters_box_layout.addWidget(role_box3)
        filters_box_content.setLayout(filters_box_layout)
        self.vision_content = QWidget()
        self.vision_layout = QVBoxLayout()
        self.vision_label3 = QLabel()
        self.vision_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vision_layout.addWidget(self.vision_label3)
        self.vision_content.setLayout(self.vision_layout)
        layout.addWidget(subtitle)
        layout.addWidget(filters_legend_content)
        layout.addWidget(filters_box_content)
        layout.addWidget(self.vision_label3)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.setLayout(layout)
        return content
    
    def AboutUsPage(self):
        image_scale = min(
            (pyautogui.size()[0] * 0.3) / 1920,
            (pyautogui.size()[1] * 0.3) / 1080,
        )
        font_scale = min(
            int((pyautogui.size()[0] * 20) / 1920),
            int((pyautogui.size()[1] * 20) / 1080),
        )
        button_font_scale = min(
            int((pyautogui.size()[0] * 20) / 1920),
            int((pyautogui.size()[1] * 20) / 1080),
        )
        button_height_scale = min(
            int((pyautogui.size()[0] * 50) / 1920),
            int((pyautogui.size()[1] * 50) / 1080),
        )
        layout = QVBoxLayout()
        content = QWidget()
        subtitle = QLabel("About Us")
        subtitle.setFont(QFont("Times New Roman", font_scale, QFont.Weight.Bold))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        # Author Image
        description_layout = QHBoxLayout()
        description_content = QWidget()
        author_image_label = QLabel()
        author_image = QPixmap(resource_path(os.path.join('assets', 'rafael_coelho_1.jpeg')))  # Replace with the path to your image
        author_image_label.setPixmap(author_image.scaled(
            int(author_image.width() * image_scale), 
            int(author_image.height() * image_scale), 
            Qt.AspectRatioMode.KeepAspectRatio))
        author_image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Author Description
        author_description = QLabel(
            "Rafael Coelho is a Brazilian Mathematics student who is "
            "passionated for Data Science and Artificial Intelligence "
            f"and works in both areas for over {str(dt.datetime.now().year - 2020)} years, with solid "
            "knowledge in technologic areas such as Machine Learning, "
            "Deep Learning, Data Science, Computer Vision, Reinforcement "
            "Learning, NLP and others.\n\n"
            "Recently, he worked in one of the Big Four companies for over a year."
        )
        author_description.setFont(QFont("Sans Serif", font_scale))
        author_description.setWordWrap(True)
        author_description.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        description_layout.addWidget(author_image_label)
        description_layout.addWidget(author_description, 1)
        description_content.setLayout(description_layout)
        # Buttons
        btn_portfolio = QPushButton('Portfolio')
        btn_linkedin = QPushButton('LinkedIn')
        btn_github = QPushButton('GitHub')
        btn_portfolio.setFont(QFont("Times New Roman", button_font_scale, QFont.Weight.Bold))
        btn_linkedin.setFont(QFont("Times New Roman", button_font_scale, QFont.Weight.Bold))
        btn_github.setFont(QFont("Times New Roman", button_font_scale, QFont.Weight.Bold))
        btn_portfolio.setFixedHeight(button_height_scale)
        btn_linkedin.setFixedHeight(button_height_scale)
        btn_github.setFixedHeight(button_height_scale)
        btn_portfolio.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://rafaelcoelho.streamlit.app/"))
        )
        btn_linkedin.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://linkedin.com/in/rafaelcoelho1409"))
        )
        btn_github.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/rafaelcoelho1409/"))
        )
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(btn_portfolio)
        buttons_layout.addWidget(btn_linkedin)
        buttons_layout.addWidget(btn_github)
        #OTHER COELHO PROJECTS
        other_projects = QLabel("Other projects")
        other_projects.setFont(QFont("Times New Roman", font_scale, QFont.Weight.Bold))
        other_projects.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
        coelho_legends = QHBoxLayout()
        coelho_finance_legend = QPushButton("COELHO Finance")
        f1_analytics_legend = QPushButton("Formula 1 Analytics")
        coelho_vision_legend = QPushButton("COELHO VISION")
        coelho_finance_legend.clicked.connect(
             lambda: QDesktopServices.openUrl(QUrl("https://coelhofinance.streamlit.app/"))
        )
        f1_analytics_legend.clicked.connect(
             lambda: QDesktopServices.openUrl(QUrl("https://f1analytics.streamlit.app/"))
        )
        coelho_vision_legend.clicked.connect(
             lambda: QDesktopServices.openUrl(QUrl("https://coelhovision.streamlit.app/"))
        )
        for x in [coelho_finance_legend, f1_analytics_legend, coelho_vision_legend]:
            x.setFont(QFont("Times New Roman", button_font_scale))
            coelho_legends.addWidget(x)
        coelho_logos = QHBoxLayout()
        coelho_finance_logo = QLabel()
        f1_analytics_logo = QLabel()
        coelho_vision_logo = QLabel()
        coelho_finance_pixmap = QPixmap(resource_path(os.path.join('assets', 'coelho_finance_logo.png')))
        f1_analytics_pixmap = QPixmap(resource_path(os.path.join('assets', 'f1_analytics_logo.jpg')))
        coelho_vision_pixmap = QPixmap(resource_path(os.path.join('assets', 'coelho_vision_logo.png')))
        coelho_finance_logo.setPixmap(coelho_finance_pixmap.scaled(
                int(300 * 0.75), 
                int(400 * 0.75), 
                Qt.AspectRatioMode.KeepAspectRatio))
        f1_analytics_logo.setPixmap(f1_analytics_pixmap.scaled(
                int(300 * 0.75), 
                int(400 * 0.75), 
                Qt.AspectRatioMode.KeepAspectRatio))
        coelho_vision_logo.setPixmap(coelho_vision_pixmap.scaled(
                int(300 * 0.75), 
                int(400 * 0.75), 
                Qt.AspectRatioMode.KeepAspectRatio))
        for x in [coelho_finance_logo, f1_analytics_logo, coelho_vision_logo]:
            x.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)
            coelho_logos.addWidget(x)
        layout.addWidget(subtitle)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._divider())
        layout.addWidget(description_content)
        layout.addLayout(buttons_layout)
        layout.addWidget(self._divider())
        layout.addWidget(other_projects)
        layout.addLayout(coelho_logos)
        layout.addLayout(coelho_legends)
        content.setLayout(layout)
        return content


    def _divider(self):
        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        h_line.setFrameShadow(QFrame.Shadow.Sunken)
        h_line.setStyleSheet("color: #c0c0c0")
        return h_line
     
    def _available_cameras(self, i):
        global AVAILABLE_CAMERA, CAP
        AVAILABLE_CAMERA = i
        CAP = cv2.VideoCapture(int(AVAILABLE_CAMERA))

    def _available_frame_sizes(self):
        self.supported_resolutions = ["-"]
        for width, height in self.resolutions:
            try:
                # Set the resolution
                global CAP
                CAP.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                # Read the actual set resolution
                actual_width = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if (actual_width, actual_height) == (width, height):
                    self.supported_resolutions.append((int(actual_width), int(actual_height)))
            except:
                pass
        return self.supported_resolutions

    def _role(self, role):
        if role == "-":
            self.model = NoModel()
        elif role == "Full Face Detection":
            self.model = FullFaceDetector()
        elif role == "Object Detection (MediaPipe)":
            self.model = MediaPipeObjectDetection()
        elif role == "Face Detector (MediaPipe)":
            self.model = MediaPipeFaceDetector()    
        elif role == "Image Segmentation (MediaPipe)":
            self.model = MediaPipeImageSegmentation()
        elif role == "Hand Landmarker (MediaPipe)":
            self.model = MediaPipeHandLandmarker()
        elif role == "Pose Estimation (MediaPipe)":
            self.model = MediaPipePoseEstimation()
        elif role == "Pose Estimation (OpenVINO)":
            self.model = OpenVINOPoseEstimation()
        elif role == "Object Detection (RoboFlow)":
            self.model = RFObjectDetection()
        elif role == "Object Tracking (RoboFlow)":
            self.model = RFTracking()
        elif role == "Semantic Segmentation - Mask Mode (OpenVINO)":
            self.model = OpenVINOImageSegmentation(True)
        elif role == "Semantic Segmentation (OpenVINO)":
            self.model = OpenVINOImageSegmentation(False)
        elif role == "Depth Estimation (OpenVINO)":
            self.model = OpenVINODepthEstimation()
        elif role == "Optical Character Recognition (OpenVINO)":
            self.model = OpenVINOOCR()
        else:
            self.model = NoModel()

    def _update_vision(self, full_screen):
        resize_rate = min(
            pyautogui.size()[0] / self.rgb_original_image.shape[1],
            pyautogui.size()[1] / self.rgb_original_image.shape[0])
        if not full_screen:
            self.rgb_displayed_image = cv2.resize(
                self.rgb_original_image, 
                (
                    int(self.rgb_original_image.shape[1] * resize_rate * 0.5),
                    int(self.rgb_original_image.shape[0] * resize_rate * 0.5)
                    ))
            # Convert the image to a format PyQt can work with
            h, w, ch = self.rgb_displayed_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                self.rgb_displayed_image.data, 
                w, 
                h, 
                bytes_per_line, 
                QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(
                w, 
                h, 
                Qt.AspectRatioMode.KeepAspectRatioByExpanding)
            # Display the image on the label
            self.vision_label1.setVisible(True)
            self.vision_label2.setVisible(True)
            self.vision_label3.setVisible(True)
            self.vision_label1.setPixmap(QPixmap.fromImage(p))
            self.vision_label2.setPixmap(QPixmap.fromImage(p))
            self.vision_label3.setPixmap(QPixmap.fromImage(p))
            self.vision_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vision_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vision_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            # Convert the image to a format PyQt can work with
            self.original_displayed_image = cv2.resize(
                self.rgb_original_image, 
                (
                    int(self.rgb_original_image.shape[1] * resize_rate * 0.9),
                    int(self.rgb_original_image.shape[0] * resize_rate * 0.9)
                    ))
            h, w, ch = self.original_displayed_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                self.original_displayed_image.data, 
                w, 
                h, 
                bytes_per_line, 
                QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(
                w, 
                h, 
                Qt.AspectRatioMode.KeepAspectRatioByExpanding)
            self.vision_label1.setVisible(False)
            self.vision_label2.setVisible(False)
            self.vision_label3.setVisible(False)
            self.full_screen_window.rgb_original_image = self.rgb_original_image
            self.full_screen_window.video_label.setPixmap(QPixmap.fromImage(p))
            self.full_screen_window.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _update_frame(self):
        self.resolution_box.setCurrentIndex(self.resolution_box.findText(CURRENT_RESOLUTION))
        ret, frame = CAP.read()
        if ret:
            frame = self.model.transform(frame)
            self.rgb_original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._update_vision(FULL_SCREEN_MODE)

    def _set_resolution(self, resolution):
        res_dict = {
            "640 x 480": (640, 480),
            "800 x 600": (800, 600),
            "1024 x 768": (1024, 768),
            "1280 x 720": (1280, 720),
            "1920 x 1080": (1920, 1080)
        }
        try:
            self.actual_resolution = res_dict[resolution]
            global RES_WIDTH, RES_HEIGHT, CAP, CURRENT_RESOLUTION
            CURRENT_RESOLUTION = resolution
            self.resolution_box.setCurrentIndex(self.resolution_box.findText(CURRENT_RESOLUTION))
            RES_WIDTH = self.actual_resolution[0]
            RES_HEIGHT = self.actual_resolution[1]
            # Set the resolution
            CAP.release()
            CAP = cv2.VideoCapture(int(self.available_cameras_box.currentText()))
            CAP.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
            CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
        except:
            pass

    def _open_full_screen(self):
        global FULL_SCREEN_MODE
        FULL_SCREEN_MODE = True
        self.full_screen_window = FullScreenWindow()
        self.full_screen_window.show()

    def _save_image(self):
        global CAP, SAVE_COUNT
        CAP.release()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "image" + f"{SAVE_COUNT:0>4}", "JPEG (*.jpg *.jpeg);;PNG (*.png)")
        print(filePath)
        if filePath:
            if ".png" in filePath:
                cv2.imwrite(filePath, cv2.cvtColor(self.rgb_original_image, cv2.COLOR_BGR2RGB))
            elif ".jpg" in filePath or ".jpeg" in filePath:
                cv2.imwrite(filePath, cv2.cvtColor(self.rgb_original_image, cv2.COLOR_BGR2RGB))
            SAVE_COUNT += 1
        CAP = cv2.VideoCapture(int(self.available_cameras_box.currentText()))
        CAP.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
        CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)

#-----------------------------------------------------------------------------
        
class FullScreenWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.showFullScreen()
        self.setWindowTitle('COELHO VISION (Full Screen)')        
        # Layout
        self.main_layout = QVBoxLayout()
        self.filter_layout = QHBoxLayout()
        self.resolution_legend = QLabel("Frame resolution")
        self.resolution_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.actual_resolution = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        self.resolution_box = QComboBox()
        self.resolution_box.addItems([
            "640 x 480",
            "800 x 600",
            "1024 x 768",
            "1280 x 720",
            "1920 x 1080"
        ])
        self.resolution_box.currentTextChanged.connect(self._set_resolution)
        self.resolution_box.setCurrentIndex(self.resolution_box.findText(CURRENT_RESOLUTION))
        self.close_full_screen_btn = QPushButton("Close Full Screen")
        self.close_full_screen_btn.clicked.connect(self._close_video_stream)
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self._save_image)
        self.video_label = QLabel()
        for x in [
            self.resolution_legend,
            self.resolution_box,
            self.save_image_btn,
            self.close_full_screen_btn
        ]:
            self.filter_layout.addWidget(x)
        self.main_layout.addLayout(self.filter_layout)
        self.main_layout.addWidget(self._divider())
        self.main_layout.addWidget(self.video_label)
        self.setLayout(self.main_layout)

    def _close_video_stream(self):
        global FULL_SCREEN_MODE
        FULL_SCREEN_MODE = False
        self.close()

    def _save_image(self):
        global CAP, SAVE_COUNT
        CAP.release()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "image" + f"{SAVE_COUNT:0>4}", "JPEG (*.jpg *.jpeg);;PNG (*.png)")
        if filePath:
            if ".png" in filePath:
                cv2.imwrite(filePath, cv2.cvtColor(self.rgb_original_image, cv2.COLOR_BGR2RGB))
            elif ".jpg" in filePath or ".jpeg" in filePath:
                cv2.imwrite(filePath, cv2.cvtColor(self.rgb_original_image, cv2.COLOR_BGR2RGB))
            SAVE_COUNT += 1
        CAP = cv2.VideoCapture(int(AVAILABLE_CAMERA))
        CAP.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
        CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)

    def _set_resolution(self, resolution):
        res_dict = {
            "640 x 480": (640, 480),
            "800 x 600": (800, 600),
            "1024 x 768": (1024, 768),
            "1280 x 720": (1280, 720),
            "1920 x 1080": (1920, 1080)
        }
        try:
            self.actual_resolution = res_dict[resolution]
            global RES_WIDTH, RES_HEIGHT, CAP, CURRENT_RESOLUTION
            CURRENT_RESOLUTION = resolution
            self.resolution_box.setCurrentIndex(self.resolution_box.findText(CURRENT_RESOLUTION))
            RES_WIDTH = self.actual_resolution[0]
            RES_HEIGHT = self.actual_resolution[1]
            # Set the resolution
            CAP.release()
            CAP = cv2.VideoCapture(int(self.available_cameras_box.currentText()))
            CAP.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
            CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
        except:
            pass

    def _divider(self):
        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        h_line.setFrameShadow(QFrame.Shadow.Sunken)
        h_line.setStyleSheet("color: #c0c0c0")
        return h_line

#-------------------------
window = MainApp()
window.show()
app.exec()

#py -m PyInstaller --onefile --windowed --clean --icon=assets/coelho_vision_icon.ico --add-data "data;data" --add-data "assets;assets" --add-data "models;models" --collect-all openvino --collect-all torch .\COELHO_VISION.py
#ALWAYS DELETE *.spec, __pycache__, build AND dist FOLDERS BEFORE COMPILING IT
    
#py -m nuitka --standalone --windows-icon-from-ico=assets/coelho_vision_icon.ico --include-data-dir=data=data --include-data-dir=assets=assets --include-data-dir=models=models COELHO_VISION.py
