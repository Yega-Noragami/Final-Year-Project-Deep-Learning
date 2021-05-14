import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import pathlib
import glob
import matplotlib.pyplot as plt
import cv2
import sys

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import piexif

#path to saved folder 
global saved_directory


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


def run_inference(model, category_index, image_path):
    
    if os.path.isdir(image_path):
        image_paths = []
        # for file_extension in ('*.png', '*jpg'):
        #     image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))
        for filename in os.listdir(image_path):
            if '.DS_Store' in filename:
                print('.DS_Store file')
            else:    
                final_path=image_path+'/'+filename
                image_paths.append(final_path)
        
        
        for i_path in image_paths:
            print('image path:', i_path)
            image_np = load_image_into_numpy_array(i_path)
            image = Image.open(i_path)
            image_width ,image_height= image.size
            print('width:',image_width,'height',image_height)
            image_name=os.path.split(i_path)[-1]
            print('image name:',image_name)
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.imshow(image_np)
            # This is the way I'm getting my coordinates
            boxes = output_dict['detection_boxes']
            # get all boxes from an array
            max_boxes_to_draw = boxes.shape[0]
            # get scores to get a threshold
            scores = output_dict['detection_scores']
            # this is set as a default but feel free to adjust it to your needs
            min_score_thresh=.5
            # iterate over all objects found   
            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                try:
                    if scores is None or scores[i] > min_score_thresh:
                        # boxes[i] is the box which will be drawn
                        borders =boxes[i]
                        class_name = category_index[output_dict['detection_classes'][i]]['name']
                        print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
                        
                        ymin = int((borders[0]*image_height))
                        xmin = int((borders[1]*image_width))
                        ymax = int((borders[2]*image_height))
                        xmax = int((borders[3]*image_width))
                        if xmax>ymax:
                            length = xmax - xmin
                            displacement = (ymax-ymin)
                            ymin = ymin - displacement
                            ymax = ymax - displacement
                            # xmin = xmin - (xmax-xmin)
                            y_length = ymin+length 
                            cropped_img = image.crop((xmin , ymin , xmax , y_length))
                            print("x is bigger in this case ")
                        elif ymax>xmax:
                            length = ymax-ymin 
                            displacement = (xmax-xmin)
                            xmin = xmin -displacement
                            xmax = xmax -displacement
                            x_length = xmin + length 
                            cropped_img= image.crop((xmin , ymin , x_length ))
                            print("y is bigger in this case ")
                        # cropped_img = image.crop((xmin,ymin,xmax,ymax))
                    

                        #declare new image new , make it easier to transfer exif data 
                        temp_image_name= image_name[:23]+'-'+str(i+1)+image_name[-4:]
                        cropped_img = cropped_img.save(saved_directory+temp_image_name)
                        print('saved location:',saved_directory+temp_image_name)

                        #save meta-tags  transplant 
                        piexif.transplant(i_path,saved_directory+temp_image_name)
                except:
                    print("error hai behen ke lode")

        
    else:
        image_np = load_image_into_numpy_array(image_path)
    
        image = Image.open(image_path)
        print('image path:', image_path)
        image_width ,image_height= image.size
        print('width:',image_width,'height',image_height)
        image_name=os.path.split(image_path)[-1]
        print('image name:',image_name)
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.imshow(image_np)
        # This is the way I'm getting my coordinates
        boxes = output_dict['detection_boxes']
        # get all boxes from an array
        max_boxes_to_draw = boxes.shape[0]
        # get scores to get a threshold
        scores = output_dict['detection_scores']
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh=.5
        # iterate over all objects found   
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                borders =boxes[i]
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])

                ymin = int((borders[0]*image_height))
                xmin = int((borders[1]*image_width))
                ymax = int((borders[2]*image_height))
                xmax = int((borders[3]*image_width))
                cropped_img = image.crop((xmin,ymin,xmax,ymax))
                if xmax>ymax:
                    length = xmax - xmin
                    displacement = (ymax-ymin)
                    ymin = ymin - displacement
                    ymax = ymax - displacement
                    # xmin = xmin - (xmax-xmin)
                    y_length = ymin+length 
                    cropped_img = image.crop((xmin , ymin , xmax , y_length))
                    print("x is bigger in this case ")
                elif ymax>xmax:
                    length = ymax-ymin 
                    displacement = (xmax-xmin)
                    xmin = xmin -displacement
                    xmax = xmax -displacement
                    x_length = xmin + length 
                    cropped_img= image.crop((xmin , ymin , x_length ))
                    print("y is bigger in this case ")
                    
                # print('saved location:',saved_directory+image_name)

                #declare new image new , make it easier to transfer exif data 
                temp_image_name= image_name[:23]+'-'+str(i+1)+image_name[-4:]
                cropped_img = cropped_img.save(saved_directory+temp_image_name)

                #save meta-tags  transplant 
                piexif.transplant(image_path,saved_directory+temp_image_name)



        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to output folder')
    
    args = parser.parse_args()

    saved_directory = args.output_path+"/"
    os.mkdir(args.output_path)
	
    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    run_inference(detection_model, category_index, args.image_path)


'''
python3 detect_image.py -m saved_model -l label_map.pbtxt -i 20201116-125113IMG_8313.jpg 
python Crop_images_exif_multiple.py -m saved_model -l label_map.pbtxt -i one -o one_copy
'''
