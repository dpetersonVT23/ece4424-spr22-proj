# David Peterson - ECE 4424 - Spring 2022 Project - AMLPR
# licenseplate.py

# import statements
import argparse
import os
import sys
import cv2
import pytesseract
import difflib
import numpy as np
import time


# main function
def main():
    # determine if commandline arguments are valid
    if args.image_filename and args.video_filename:                         # if both an image file name and video file name are specified in arguments
        sys.exit("ERROR: More than one file type specified. Exiting...")        # exit program with error message

    if args.image_filename:                                                 # if image file name specified for analysis
        if not os.path.exists(args.image_filename):                             # if specified image file name does not exist
            sys.exit("ERROR: Invalid image file name specified. Exiting...")        # exit program with error message

        # program status update
        print("[STATUS] Processing {}...".format(args.image_filename), end="", flush=True)

        curr_img = cv2.imread(args.image_filename)  # open specified image
        proc_time_start = time.time()               # start processing timer

        detections = detectLicensePlate(curr_img)                       # detect license plate location(s) in current image
        lps = []                                                        # list of predicted license plate numbers in specified image
        if len(detections):                                             # if license plate detection successful on specified image
            for detection in detections:                                    # for each detected license plate
                prediction = recognizeLicensePlate(detection)                   # attempt to recognize license plate number

                if len(prediction):                                             # if attempt to recognize license plate number successful
                    if checkUniqueness(lps, prediction):                            # if detected license plate number has not been detected previously in image
                        lps.append(filterLicensePlate(prediction))                      # append predicted license plate numbers

        # end processing timer and calculate processing time
        proc_time_end = time.time()
        proc_time = round(proc_time_end - proc_time_start, 3)

        # program status update
        print("Done!")

        # program results
        print("[RESULT] Processing time for {}: {} seconds".format(args.image_filename, proc_time))
        print("[RESULT] Number of detected license plates in {}: {}".format(args.image_filename, len(lps)))
        print("[RESULT] Predictions of recognized license plates: {}".format(lps))

        # prompt user to request visualization of detections
        visualize = input("[REQUEST] Would you like to visualize detections? [Y/n]: ")  # prompt user input
        if visualize == 'Y':                                                            # if user requests visualization of detections
            cv2.imshow("Original Input Image", cv2.resize(curr_img, (600, 800)))            # display resized original input image
            for idx, detection in enumerate(detections):                                    # for each detected license plate
                cv2.imshow("Detection #" + str(idx+1), cv2.resize(detection, (400, 200)))       # display resized cropped license plate
            cv2.waitKey()                                                                   # wait for user to exit windows
    elif args.video_filename:                                               # else if video file name specified for analysis
        if not os.path.exists(args.video_filename):                             # if specified image file name does not exist
            sys.exit("ERROR: Invalid video file name specified. Exiting...")        # exit program with error message

        # program status update
        print("[STATUS] Processing {}...".format(args.video_filename), end="", flush=True)

        vc = cv2.VideoCapture(args.video_filename)                          # open specified video capture
        vc_fps = round(vc.get(cv2.CAP_PROP_FPS))                            # get video capture frames per second - rounded to nearest integer
        lps = []                                                            # list of predicted license plate numbers in video capture

        read_status, curr_vcf = vc.read()                                   # initialize video capture read status and first video capture frame
        frame_count = 0                                                     # initialize video capture frame count
        while read_status:                                                  # while video capture frame read successfully
            if not frame_count % vc_fps:                                        # process every 1 seconds in video capture
                detections = detectLicensePlate(curr_vcf)                           # detect license plate location(s) in current video capture frame

                if len(detections):                                                 # if license plate detection successful on specified image
                    for detection in detections:                                        # for each detected license plate
                        prediction = recognizeLicensePlate(detection)                       # attempt to recognize license plate number

                        if len(prediction):                                                 # if attempt to recognize license plate number successful
                            if checkUniqueness(lps, prediction):                                # if detected license plate number has not been detected previously in image
                                lps.append(filterLicensePlate(prediction))                          # append predicted license plate numbers

            read_status, curr_vcf = vc.read()           # get next video capture frame
            frame_count += 1                            # increment read video capture frame count

        # program status update
        print("Done!")

        # program results
        print("[RESULT] Number of detected license plates in video: {}".format(len(lps)))
        print("[RESULT] Predictions of recognized license plates: {}".format(lps))
    else:                                                                   # else
        sys.exit("ERROR: No file specified for analysis. Exiting...")           # exit program with error message

    return 0    # program executed successfully


# function to detect license plate location(s) in the given image/video capture frame
# param curr - current unmodified image/video capture frame
# return true if license plate(s) detected in the given image/video capture frame, otherwise false
# return cropped version of given image/video capture frame
def detectLicensePlate(curr):
    curr_w, curr_h = curr.shape[1], curr.shape[0]  # extract the width and height of the current image/video capture frame

    # execute forward pass of YOLO license plate detector with blob from the current input image
    # obtain bounding boxes, associated probabilities, and processing time
    scale = 1 / 255.0                                                               # set the scale
    blob = cv2.dnn.blobFromImage(curr, scale, (416, 416), swapRB=True, crop=False)  # create blob from input image
    net.setInput(blob)                                                              # set input to the network

    outputs = net.forward(output_layer_names)  # get output from the network

    # process results of YOLOv3 model outputs for the current image
    boxes = []                  # list to store detected bounding boxes
    confidences = []            # list to store confidences of detected bounding boxes
    class_IDs = []              # list to store class ID of detected bounding boxes
    confidence_thresh = 0.7     # confidence threshold
    nms_thresh = 0.3            # non-maxima suppression threshold

    for output in outputs:                                          # for each output
        for detection in output:                                        # for each detection in the current output
            scores = detection[5:]                                          # obtain the scores for the current detection
            class_ID = np.argmax(scores)                                    # extract the class ID from the scores
            confidence = scores[class_ID]                                   # extract the confidence/probability from the scores

            if confidence > confidence_thresh:                              # if confidence of detection is above confidence threshold
                box = detection[0:4] * np.array([curr_w, curr_h,
                                                 curr_w, curr_h])               # scale the bounding box coordinates to current input image size
                cx, cy, w, h = box.astype("int")                                # extract bounding box information => center (x,y) coordinates, width, and height
                tlx, tly = int(cx - (w / 2)), int(cy - (h / 2))                 # compute top-left (x,y) coordinates from center (x,y) coordinates

                boxes.append([tlx, tly, int(w), int(h)])                        # update list of detected bounding boxes
                confidences.append(float(confidence))                           # update list of confidences
                class_IDs.append(class_ID)                                      # update list of class IDs

    # apply non-maxima suppression (NMS), eliminate weak + overlapping bounding boxes
    detections = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)

    # crop detections on current input image
    curr_cropped_detections = []                                                    # initialize list to store all cropped license plate images
    if len(detections):                                                             # if there is at least 1 license plate detection
        for i in detections.flatten():                                                  # for each kept index
            x, y = boxes[i][0], boxes[i][1]                                                 # extract bounding box top-left (x,y) coordinates
            w, h = boxes[i][2], boxes[i][3]                                                 # extract bounding box width and height

            wbuf = -1                                                                       # width cropping buffer to ensure license plate characters are not cut off
            hbuf = -1                                                                       # height cropping buffer to ensure license plate characters are not cut off
            curr_cropped_detections.append(curr[y-hbuf:y+h+hbuf, x-wbuf:x+w+wbuf])          # crop the current input image to the detected bounding box

    return curr_cropped_detections  # return cropped image detections


# function to perform image processing on the given image/video capture frame
# param curr - current cropped image/video capture frame
# return processed version of given image/video capture frame
def process(curr):
    # resize current image to have more pixels/detail
    # no desired size
    # scale width and height by factor of 4
    # bicubic interpolation over 4×4 pixel neighborhood
    resize_curr = cv2.resize(curr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # apply colorspace conversion to grayscale for image thresholding
    gray_curr = cv2.cvtColor(resize_curr, cv2.COLOR_BGR2GRAY)

    # apply weighted gaussian blur filter - remove image noise
    # 5x5 kernel size - size directly proportional to amount of blur
    # sigma set to 0 - automatically compute sigma based on kernel size
    gauss_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)

    # apply image thresholding (binarization - convert all pixels to black or white)
    # 0 for threshold value - OTSU threshold used instead
    # 255 (white) is new pixel value if current pixel value less than OTSU threshold
    bin_curr = cv2.threshold(gauss_curr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply image dilation to remove image noise after binarization
    # 3x3 kernel size - size directly proportional to amount of dilation
    kernel = np.ones((3, 3), np.uint8)
    final_curr = cv2.dilate(bin_curr, kernel)

    return final_curr   # return final processed version of given image/video capture frame


# function to filter predicted license plate number
# param prediction - predicted license plate number
# return filtered predicted license plate number
def filterLicensePlate(prediction):
    lp_char_accept = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"     # list of accepted predicted license plate characters
    for char in prediction:                                     # for each character in the predicted license plate number
        if char not in lp_char_accept:                              # if character is not accepted
            prediction = prediction.replace(char, "")                   # remove the character

    return prediction   # return filtered predicted license plate number


# function to recognize license plate number(s) in the current image/video capture frame
# param curr - current cropped image/video capture frame
# return true if license plate number(s) recognized in the given image/video capture frame, otherwise false
# return predicted license plate number(s)
def recognizeLicensePlate(curr):
    curr_processed = process(curr)  # perform image processing on current image/video capture frame

    config = '--oem 3 --psm 6'                                                              # tesseract OCR configuration
    curr_prediction = ""                                                                    # initialize prediction string for current image/video capture frame
    curr_prediction_data = pytesseract.image_to_data(curr_processed,
                                                     config=config,
                                                     output_type=pytesseract.Output.DICT)   # get prediction data for current image/video capture frame

    for idx, conf in enumerate(curr_prediction_data['conf']):       # for each enumerated prediction confidence value
        if int(conf) > 0:                                               # if confidence value is greater than 0%
            curr_prediction += curr_prediction_data['text'][idx]            # concatenate prediction text to prediction string

    filtered_prediction = filterLicensePlate(curr_prediction)       # filter predicted license plate number characters

    return filtered_prediction  # return filtered predicted license plate number


# function to determine uniqueness of predicted license plate number
# param lps - previously predicted unique license plates
# param curr - current predicted license plates under comparison
# return true if license plate similarity is below unique threshold for all previously predicted license plates, otherwise false
def checkUniqueness(lps, curr):
    uniqueness_thresh = 0.85                                                        # set predicted license plate number uniqueness threshold to 85%
    for lp in lps:                                                                  # for each previously predicted license plate
        similarity_percentage = difflib.SequenceMatcher(None, lp, curr).ratio()         # use built-in string sequence matcher to determine similarity percentage
        if similarity_percentage >= uniqueness_thresh:                                  # if similarity percentage above uniqueness threshold
            return False                                                                    # current predicted license plate is not unique

    return True     # current predicted license plate is unique


if __name__ == "__main__":
    # handle pre-processing for commandline argument(s)
    # program status update
    print("[STATUS] Parsing commandline arguments...", end="", flush=True)

    parser = argparse.ArgumentParser()  # initialize instance of Argument Parser

    # add arguments to parser instance
    parser.add_argument('--video', dest='video_filename', type=str, help='Specify video file name to be analyzed.')  # video file argument
    parser.add_argument('--image', dest='image_filename', type=str, help='Specify image file name to be analyzed.')  # image file argument

    # parse arguments
    args = parser.parse_args()

    # program status update
    print("Done!")
    print("[STATUS] Loading YOLO model...", end="", flush=True)

    # load YOLO model
    # load class names files and map class names to a random color
    class_names_file = open("yolo/classes.names", "r")                          # open class names file
    class_names_colors = {}                                                     # dictionary to store class name:class color key:value pairs
    for line in class_names_file:                                               # for each class
        class_names_colors[line.strip()] = tuple(np.random.uniform(0, 255, 3))      # assign the class a randomly generated color

    # load pre-trained YOLOv3 license plate object detector
    net = cv2.dnn.readNetFromDarknet("yolo/yolov3-custom.cfg",
                                     "yolo/darknet-yolov3-train-custom_final.weights")

    # determine output layers names from YOLO architecture
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # program status update
    print("Done!")

    # call main function
    main()
