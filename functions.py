import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from urllib.request import urlretrieve
import os 
import sys
from inference import get_model
import supervision as sv
import matplotlib.pyplot as plt
import openvino as ov
import ipywidgets as widgets
import matplotlib
from numpy.lib.stride_tricks import as_strided
import time
from PIL import Image
from decoder import OpenPoseDecoder


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

############################################################################################
#COMPUTER VISION
def Hex_to_RGB(ip):
    return tuple(int(ip[i:i+2],16) for i in (0, 2, 4))

class FullFaceDetector:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            resource_path(os.path.join('data', 'haarcascade_frontalface_default.xml'))
        )
        self.eye_detector = cv2.CascadeClassifier(
            resource_path(os.path.join('data', 'haarcascade_eye.xml'))
        )
        self.mouth_detector = cv2.CascadeClassifier(
            resource_path(os.path.join('data', 'haarcascade_mcs_mouth.xml'))
        )
        self.font_scale = 0.5
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.face_scaleFactor = 1.50
        self.eye_scaleFactor = 1.30
        self.mouth_scaleFactor = 3.00
        self.face_minNeighbors = 5
        self.eye_minNeighbors = 7
        self.mouth_minNeighbors = 10
    def transform(
            self, 
            img):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_img, self.face_scaleFactor, self.face_minNeighbors)
            eyes = self.eye_detector.detectMultiScale(gray_img, self.eye_scaleFactor, self.eye_minNeighbors)
            mouths = self.mouth_detector.detectMultiScale(gray_img, self.mouth_scaleFactor, self.mouth_minNeighbors)
            for detector, color, name in [
                (faces, (255, 0, 0), "Face"), 
                (eyes, (0, 255, 0), "Eye"),
                (mouths, (0, 0, 255), "Mouth")]:
                for (x, y, w, h) in detector:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(name, self.font, fontScale = self.font_scale, thickness = 1)
                    # Create a filled rectangle for label background
                    label_background_start = (x, y - text_height - 8)  # Adjusted to position above the box
                    label_background_end = (x + text_width, y)
                    img = cv2.rectangle(img, label_background_start, label_background_end, color, thickness = cv2.FILLED)
                    img = cv2.putText(img, name, (x, y - 5), self.font, self.font_scale, (0,0,0), 1)
            return img
    

class MediaPipeObjectDetection:
    def __init__(self):
        self.text_color = (255, 0, 0)
        #Create an ObjectDetector object
        self.model_path = resource_path(os.path.join('models', 'efficientdet.tflite'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
        #    urlretrieve(url, self.model_path)
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.ObjectDetectorOptions(
            base_options = self.base_options,
            score_threshold = 0.5)
        self.detector = vision.ObjectDetector.create_from_options(self.options)
    #@numba.jit
    def visualize(self, image, detection_result):
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, self.text_color, 3)   
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        FONT_SIZE, self.text_color, FONT_THICKNESS)    
        return image
    #@numba.jit
    def transform(self, original_image):
        #Load the input image
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        #Detect objects in the input image
        detection_result = self.detector.detect(image)
        #Process the detection result
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result)
        return annotated_image


class MediaPipeImageClassification:
    def __init__(self):
        self.model_path = resource_path(os.path.join('models', 'classifier.tflite'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
        #    urlretrieve(url, self.model_path)
        #Create an ImageClassifier object
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.ImageClassifierOptions(
            base_options = self.base_options, 
            max_results = 4)
        self.classifier = vision.ImageClassifier.create_from_options(self.options)
    #@numba.jit
    def transform(self, original_image):
        #Load the input image
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        #Classify the input image
        classification_result = self.classifier.classify(image)
        #Process the classification result
        return classification_result


class MediaPipeImageSegmentation:
    def __init__(self):
        self.mask_color = "000000"
        self.background_color = "ffffff"
        self.model_path = resource_path(os.path.join('models', 'deeplabv3.tflite'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
        #    urlretrieve(url, self.model_path)
        # Create the options that will be used for ImageSegmenter
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.ImageSegmenterOptions(
            base_options = self.base_options,
            output_category_mask = True)
    def transform(self, original_image):
        # Create the image segmenter
        with vision.ImageSegmenter.create_from_options(self.options) as segmenter:
            # Create the MediaPipe image file that will be segmented
            image = mp.Image(
                image_format = mp.ImageFormat.SRGB,
                data = original_image
            )
            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = Hex_to_RGB(self.mask_color[1:])
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = Hex_to_RGB(self.background_color[1:])
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
        return output_image


class MediaPipeGestureRecognition:
    def __init__(self):
        self.model_path = resource_path(os.path.join('models', 'gesture_recognizer.task'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        #    urlretrieve(url, self.model_path)
        #Create an GestureRecognizer object
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options = self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
    #@numba.jit
    def transform(self, original_image):
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        recognition_result = self.recognizer.recognize(image)
        return recognition_result
    

class MediaPipeHandLandmarker:
    def __init__(self):
        self.model_path = resource_path(os.path.join('models', 'hand_landmarker.task'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        #    urlretrieve(url, self.model_path)
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options = self.base_options,
            num_hands = 2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
    #@numba.jit
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (255, 0, 0)
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]   
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
              annotated_image,
              hand_landmarks_proto,
              solutions.hands.HAND_CONNECTIONS,
              solutions.drawing_styles.get_default_hand_landmarks_style(),
              solutions.drawing_styles.get_default_hand_connections_style())    
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN  
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)  
        return annotated_image
    #@numba.jit
    def transform(self, original_image):
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = self.detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return annotated_image


class MediaPipeFaceDetector:
    def __init__(self):
        self.model_path = resource_path(os.path.join('models', 'detector.tflite'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        #    urlretrieve(url, self.model_path)
        self.base_options = python.BaseOptions(
            model_asset_path = resource_path(os.path.join('models', 'detector.tflite')))
        self.options = vision.FaceDetectorOptions(base_options = self.base_options)
        self.detector = vision.FaceDetector.create_from_options(self.options)
    #@numba.jit
    def _normalized_to_pixel_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value)) 
        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    #@numba.jit
    def visualize(self, image, detection_result):
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red
        annotated_image = image.copy()
        height, width, _ = image.shape  
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3) 
            # Draw keypoints
            for keypoint in detection.keypoints:
              keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                             width, height)
              color, thickness, radius = (0, 255, 0), 2, 2
              cv2.circle(annotated_image, keypoint_px, thickness, color, radius)  
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)    
        return annotated_image
    #@numba.jit
    def transform(self, original_image):
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = self.detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result)
        return annotated_image
    

class MediaPipePoseEstimation:
    def __init__(self):
        self.model_path = resource_path(os.path.join('models', 'pose_landmarker.task'))
        #if not os.path.exists(self.model_path):
        #    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        #    urlretrieve(url, self.model_path)
        self.base_options = python.BaseOptions(
            model_asset_path = self.model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options = self.base_options,
            output_segmentation_masks = True)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
    #@numba.jit
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
              annotated_image,
              pose_landmarks_proto,
              solutions.pose.POSE_CONNECTIONS,
              solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    #@numba.jit
    def transform(self, original_image):
        image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = original_image
        )
        detection_result = self.detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        return annotated_image
    

class NoModel:
    def __init__(self):
        pass
    def transform(self, original_image):
        return original_image
    

class RFObjectDetection:
    def __init__(self):
        # Load processor and model
        self.model = get_model(model_id="yolov8n-640")
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    def transform(self, original_image):
        # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
        results = self.model.infer(original_image)
        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
        # annotate the image with our inference results
        annotated_image = self.bounding_box_annotator.annotate(
            scene = original_image, detections = detections)
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections)
        return annotated_image
    
    
class OpenVINOImageSegmentation:
    def __init__(self, mask_mode = False):
        self.mask_mode = mask_mode
        #self.model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml"
        #self.model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin"
        self.model_xml_path = resource_path(os.path.join('models', 'road-segmentation-adas-0001.xml'))
        self.model_bin_path = resource_path(os.path.join('models', 'road-segmentation-adas-0001.bin'))
        #if not os.path.exists(self.model_xml_path):
        #    urlretrieve(self.model_xml_url, self.model_xml_path)
        #if not os.path.exists(self.model_bin_path):
        #    urlretrieve(self.model_bin_url, self.model_bin_path)
        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options = self.core.available_devices + ["AUTO"],
            value = 'AUTO',
            description = 'Device:',
            disabled = False,
        )
        self.model = self.core.read_model(model = self.model_xml_path)
        self.compiled_model = self.core.compile_model(
            model = self.model, 
            device_name = self.device.value,
            config = {"PERFORMANCE_HINT": "LATENCY"})
        self.input_layer_ir = self.compiled_model.input(0)
        self.output_layer_ir = self.compiled_model.output(0)
    def segmentation_map_to_image(
        self, result: np.ndarray, colormap: np.ndarray, remove_holes: bool = False
    ) -> np.ndarray:
        if len(result.shape) != 2 and result.shape[0] != 1:
            raise ValueError(
                f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
            )
        if len(np.unique(result)) > colormap.shape[0]:
            raise ValueError(
                f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
                "different output values. Please make sure to convert the network output to "
                "pixel values before calling this function."
            )
        elif result.shape[0] == 1:
            result = result.squeeze(0)
        result = result.astype(np.uint8)
        contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
        mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label_index, color in enumerate(colormap):
            label_index_map = result == label_index
            label_index_map = label_index_map.astype(np.uint8) * 255
            contours, hierarchies = cv2.findContours(
                label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                mask,
                contours,
                contourIdx=-1,
                color=color.tolist(),
                thickness=cv2.FILLED,
            )
        return mask
    def transform(self, original_image):
        original_image_h, original_image_w, _ = original_image.shape
        N, C, H, W = self.input_layer_ir.shape
        resized_image = cv2.resize(original_image, (W, H))
        # Reshape to the network input shape.
        input_image = np.expand_dims(
            resized_image.transpose(2, 0, 1), 0
        )
        # Run the inference.
        result = self.compiled_model([input_image])[self.output_layer_ir]
        # Prepare data for visualization.
        segmentation_mask = np.argmax(result, axis = 1)
        colormap = np.array([[68, 1, 84], [48, 103, 141], [53, 183, 120], [199, 216, 52]])
        # Define the transparency of the segmentation mask on the photo.
        alpha = 0.3
        mask = self.segmentation_map_to_image(segmentation_mask, colormap)
        resized_mask = cv2.resize(mask, (original_image_w, original_image_h))
        # Create an image with mask.
        image_with_mask = cv2.addWeighted(resized_mask, alpha, original_image, 1 - alpha, 0)
        if self.mask_mode == True:
            return resized_mask
        else:
            return image_with_mask

        
class OpenVINODepthEstimation:
    def __init__(self, mask_mode = False):
        #self.model_xml_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/MiDaS_small.xml'
        #self.model_bin_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/MiDaS_small.bin'
        self.model_xml_path = resource_path(os.path.join('models', 'MiDaS_small.xml'))
        self.model_bin_path = resource_path(os.path.join('models', 'MiDaS_small.bin'))
        #if not os.path.exists(self.model_xml_path):
        #    urlretrieve(self.model_xml_url, self.model_xml_path)
        #if not os.path.exists(self.model_bin_path):
        #    urlretrieve(self.model_bin_url, self.model_bin_path)
        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options = self.core.available_devices + ["AUTO"],
            value = 'AUTO',
            description = 'Device:',
            disabled = False,
        )
        self.core.set_property({'CACHE_DIR': '../cache'})
        self.model = self.core.read_model(self.model_xml_path)
        self.compiled_model = self.core.compile_model(
            model = self.model, 
            device_name = self.device.value,
            config = {"PERFORMANCE_HINT": "LATENCY"})
        self.input_key = self.compiled_model.input(0)
        self.output_key = self.compiled_model.output(0)
        self.network_input_shape = list(self.input_key.shape)
        self.network_image_height, self.network_image_width = self.network_input_shape[2:]
    def normalize_minmax(self, data):
        """Normalizes the values in `data` between 0 and 1"""
        return (data - data.min()) / (data.max() - data.min())
    def convert_result_to_image(self, result, colormap = "viridis"):
        cmap = matplotlib.cm.get_cmap(colormap)
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        return result
    def transform(self, original_image):
        # Resize to input shape for network.
        resized_image = cv2.resize(
            src = original_image, 
            dsize = (self.network_image_height, self.network_image_width))
        # Reshape the image to network input shape NCHW.
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
        result = self.compiled_model([input_image])[self.output_key]
        result_image = self.convert_result_to_image(result = result)
        result_image = cv2.resize(result_image, original_image.shape[:2][::-1])
        return result_image
    

class OpenVINOPoseEstimation:
    def __init__(self):
        #self.model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml"
        #self.model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.bin"
        self.model_xml_path = resource_path(os.path.join('models', 'human-pose-estimation-0001.xml'))
        self.model_bin_path = resource_path(os.path.join('models', 'human-pose-estimation-0001.bin'))
        #if not os.path.exists(self.model_xml_path):
        #    urlretrieve(self.model_xml_url, self.model_xml_path)
        #if not os.path.exists(self.model_bin_path):
        #    urlretrieve(self.model_bin_url, self.model_bin_path)
        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options = self.core.available_devices + ["AUTO"],
            value = 'AUTO',
            description = 'Device:',
            disabled = False,
        )
        # Read the network from a file.
        self.model = self.core.read_model(self.model_xml_path)
        # Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
        self.compiled_model = self.core.compile_model(
            model = self.model, 
            device_name = self.device.value, 
            config = {"PERFORMANCE_HINT": "LATENCY"})
        # Get the input and output names of nodes.
        self.input_layer = self.compiled_model.input(0)
        self.output_layers = self.compiled_model.outputs
        # Get the input size.
        self.height, self.width = list(self.input_layer.shape)[2:]
        self.decoder = OpenPoseDecoder()
        self.colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
                  (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
                  (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))
        self.default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
        self.pafs_output_key = self.compiled_model.output("Mconv7_stage2_L1")
        self.heatmaps_output_key = self.compiled_model.output("Mconv7_stage2_L2")
    # 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
    def pool2d(self, A, kernel_size, stride, padding, pool_mode="max"):
        # Padding
        A = np.pad(A, padding, mode = "constant")
        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
        )
        A_w = A_w.reshape(-1, *kernel_size)
        # Return the result of pooling.
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)
    # non maximum suppression
    def heatmap_nms(self, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)
    # Get poses from results.
    def process_results(self, img, pafs, heatmaps):
        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [[self.pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
        )
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)

        # Decode poses.
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = list(self.compiled_model.output(index=0).partial_shape)
        output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores
    def draw_poses(self, img, poses, point_score_threshold):
        if poses.size == 0:
            return img
        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, self.colors[i], 2)
            # Draw limbs.
            for i, j in self.default_skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=self.colors[j], thickness=4)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img
    # Main processing function to run pose estimation.
    def transform(self, frame):
        input_img = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Create a batch of images (size = 1).
        input_img = input_img.transpose((2,0,1))[np.newaxis, ...]
        # Measure processing time.
        start_time = time.time()
        # Get results.
        results = self.compiled_model([input_img])
        stop_time = time.time()
        pafs = results[self.pafs_output_key]
        heatmaps = results[self.heatmaps_output_key]
        # Get poses from network results.
        poses, scores = self.process_results(frame, pafs, heatmaps)
        # Draw poses on a frame.
        frame = self.draw_poses(frame, poses, 0.1)
        return frame


class OpenVINOOCR:
    def __init__(self):
        self.detection_model_xml_path = resource_path(os.path.join("models", "horizontal-text-detection-0001.xml"))
        self.recognition_model_xml_path = resource_path(os.path.join("models", "text-recognition-resnet-fc.xml"))
        self.detection_model_bin_path = resource_path(os.path.join("models", "horizontal-text-detection-0001.bin"))
        self.recognition_model_bin_path = resource_path(os.path.join("models", "text-recognition-resnet-fc.bin"))
        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options = self.core.available_devices + ["AUTO"],
            value = 'AUTO',
            description = 'Device:',
            disabled = False,
        )
        self.detection_model = self.core.read_model(
            model = self.detection_model_xml_path, 
            weights = self.detection_model_bin_path
        )
        self.detection_compiled_model = self.core.compile_model(
            model = self.detection_model, 
            device_name = self.device.value)
        self.detection_input_layer = self.detection_compiled_model.input(0)
        self.det_N, self.det_C, self.det_H, self.det_W = self.detection_input_layer.shape
        self.recognition_model = self.core.read_model(
            model = self.recognition_model_xml_path, 
            weights = self.recognition_model_bin_path
        )
        self.recognition_compiled_model = self.core.compile_model(
            model = self.recognition_model, 
            device_name = self.device.value)
        self.recognition_output_layer = self.recognition_compiled_model.output(0)
        self.recognition_input_layer = self.recognition_compiled_model.input(0)
        # Get the height and width of the input layer.
        _, _, self.rec_H, self.rec_W = self.recognition_input_layer.shape
    def multiply_by_ratio(self, ratio_x, ratio_y, box):
        return [
            max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
            for idx, shape in enumerate(box[:-1])
        ]
    def run_preprocesing_on_crop(self, crop, net_shape):
        temp_img = cv2.resize(crop, net_shape)
        temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
        return temp_img
    def convert_result_to_image(self, original_image, resized_image, boxes, threshold=0.3, conf_labels=True):
        # Define colors for boxes and descriptions.
        colors = {"red": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}
        # Fetch image shapes to calculate a ratio.
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
        # Iterate through non-zero boxes.
        for box, annotation in boxes:
            # Pick a confidence factor from the last place in an array.
            conf = box[-1]
            if conf > threshold:
                # Convert float to int and multiply position of each box by x and y ratio.
                (x_min, y_min, x_max, y_max) = map(int, self.multiply_by_ratio(ratio_x, ratio_y, box))
                # Draw a box based on the position. Parameters in the `rectangle` function are: image, start_point, end_point, color, thickness.
                cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
                # Add a text to an image based on the position and confidence. Parameters in the `putText` function are: image, text, bottomleft_corner_textfield, font, font_scale, color, thickness, line_type
                if conf_labels:
                    # Create a background box based on annotation length.
                    (text_w, text_h), _ = cv2.getTextSize(
                        f"{annotation}", cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1
                    )
                    image_copy = original_image.copy()
                    cv2.rectangle(
                        image_copy,
                        (x_min, y_min - text_h - 10),
                        (x_min + text_w, y_min - 10),
                        colors["white"],
                        -1,
                    )
                    # Add weighted image copy with white boxes under a text.
                    cv2.addWeighted(image_copy, 0.4, original_image, 0.6, 0, original_image)
                    cv2.putText(
                        original_image,
                        f"{annotation}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        colors["red"],
                        1,
                        cv2.LINE_AA,
                    )
        return original_image
    def transform(self, original_image):
        # Resize the image to meet network expected input sizes.
        resized_image = cv2.resize(original_image, (self.det_W, self.det_H))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        output_key = self.detection_compiled_model.output("boxes")
        boxes = self.detection_compiled_model([input_image])[output_key]
        # Remove zero only boxes.
        boxes = boxes[~np.all(boxes == 0, axis=1)]
        # Calculate scale for image resizing.
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
        # Convert the image to grayscale for the text recognition model.
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # Get a dictionary to encode output, based on the model documentation.
        letters = "~0123456789abcdefghijklmnopqrstuvwxyz"
        # Prepare an empty list for annotations.
        annotations = list()
        cropped_images = list()
        # fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
        # Get annotations for each crop, based on boxes given by the detection model.
        for i, crop in enumerate(boxes):
            # Get coordinates on corners of a crop.
            (x_min, y_min, x_max, y_max) = map(int, self.multiply_by_ratio(ratio_x, ratio_y, crop))
            image_crop = self.run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (self.rec_W, self.rec_H))
            # Run inference with the recognition model.
            result = self.recognition_compiled_model([image_crop])[self.recognition_output_layer]
            # Squeeze the output to remove unnecessary dimension.
            recognition_results_test = np.squeeze(result)
            # Read an annotation based on probabilities from the output layer.
            annotation = list()
            for letter in recognition_results_test:
                parsed_letter = letters[letter.argmax()]
                # Returning 0 index from `argmax` signalizes an end of a string.
                if parsed_letter == letters[0]:
                    break
                annotation.append(parsed_letter)
            annotations.append("".join(annotation))
            cropped_image = Image.fromarray(original_image[y_min:y_max, x_min:x_max])
            cropped_images.append(cropped_image)
        boxes_with_annotations = list(zip(boxes, annotations))
        try:
            return self.convert_result_to_image(original_image, resized_image, boxes_with_annotations, conf_labels=True)
        except:
            return original_image

class RFTracking:
    def __init__(self):
        self.model = get_model(model_id = "yolov8n-640")
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def transform(self, original_image):
        try:
            results_ = self.model.infer(original_image)
            results = results_[0]
            detections = sv.Detections.from_inference(results)
            detections = self.tracker.update_with_detections(detections)
            labels = [
                f"#{tracker_id} {results.predictions[class_id].class_name}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]
            annotated_frame = self.box_annotator.annotate(
                original_image.copy(), 
                detections = detections)
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels)
            return self.trace_annotator.annotate(
                annotated_frame, detections=detections)
        except:
            return original_image





    




