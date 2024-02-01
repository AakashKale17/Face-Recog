import cv2
import numpy as np
import time
import os
import sys
import spotipy
import pandas as pd
import copy
import random
import webview

from cv2 import dnn
from math import ceil
from tkinter import *
from PIL import ImageTk, Image
from nicegui import ui
from spotipy.oauth2 import SpotifyClientCredentials

cid = 'fd8663f99b69484582989cbf717c7197'
secret = '03373799a57740daa9493d0abc30acde'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0]
]
strides = [8.0, 16.0, 32.0, 64.0]
threshold = 0.5


def main():
    w = Tk()

    w.geometry("1350x720")
    w.configure(bg="#141414")
    w.title("Facial Emotion Recognition")
    w.resizable(width=True, height=True)

    def bttn(x, y, text, bcolor, fcolor, cmd):
        def on_enter(e):
            mybutton['background'] = bcolor
            mybutton['foreground'] = fcolor

        def on_leave(e):
            mybutton['background'] = fcolor
            mybutton['foreground'] = bcolor

        mybutton = Button(w, width=42, height=2, text=text, fg=bcolor, bg=fcolor, border=0, activebackground=bcolor,
                          activeforeground=fcolor, command=cmd, )
        mybutton.place(x=x, y=y)
        mybutton.bind("<Enter>", on_enter)
        mybutton.bind("<Leave>", on_leave)

    x = "Add a heading (1).png"
    img = Image.open(x)
    img = img.resize((1370, 720))
    img = ImageTk.PhotoImage(img)
    panel = Label(w, image=img)
    panel.image = img
    panel.place(x=-2, y=-2)

    bttn(150, 400, "S T A R T", "#ffcc66", "#141414", FER_live_cam)
    bttn(150, 500, "A N A L Y S I S", "#ffcc66", "#141414", analysis)
    bttn(150, 600, "E X I T", "#ffcc66", "#141414", w.destroy)

    w.mainloop()

def analysis():
    emotion = []
    neutral = []
    happiness = []
    surprise = []
    sadness = []
    anger = []
    disgust = []
    fear = []

    with open('emoseq.txt', 'r') as f:
        for line in f:
            emotion.append(line.replace("\n",""))
    
    for i in emotion:
        if i == "neutral":
            neutral.append(i)
        elif i == "happiness":
            happiness.append(i)
        elif i == "surprise":
            surprise.append(i)
        elif i == "sadness":
            sadness.append(i)
        elif i == "anger":
            anger.append(i)
        elif i == "disgust":
            disgust.append(i)
        elif i == "fear":
            fear.append(i)

    dominant = [len(neutral),len(happiness),len(surprise),len(sadness),len(anger),len(disgust),len(fear)]

    match dominant.index(max(dominant)):
        case 0:
            get_neutral()
        case 1:
            get_happiness()
        case 2:
            get_surprise()
        case 3:
            get_sadness()
        case 4:
            get_anger()
        case 5:
            get_disgust()
        case 6:
            get_fear()

def get_html(song_id,text1,text2,text3,text4):
    html = """
        <html>
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">
                <style type="text/css">
                    body {
                            font-family: 'Montserrat', sans-serif;
                            background: rgb(40,40,40);
                            background: linear-gradient(142deg, rgba(40,40,40,1) 0%, rgba(31,31,31,1) 100%);
                        }
                    h1 {
                            color: #EEEEEE;
                    }
                    h3 {
                            color: #bfbfbf;
                    }                    
                    #card1{
                            background: rgb(40,40,40);
                            background: linear-gradient(142deg, rgba(40,40,40,1) 0%, rgba(31,31,31,1) 100%);
                            width: 550px;
                            height: 350px;
                            margin: 10px;
                            border-radius: 10px;
                            transition: transform .2s;
                            padding: 20px;
                    }
                    #container{
                            display: flex;
                    }
                    #card2{
                            background: rgb(40,40,40);
                            background: linear-gradient(142deg, rgba(40,40,40,1) 0%, rgba(31,31,31,1) 100%);
                            width: 200px;
                            margin: 10px;
                            border-radius: 10px;
                            transition: transform .2s;
                            padding: 10px;    
                    }
                    iframe{
                            transition: transform .2s;
                    }
                    iframe:hover{
                            box-shadow: rgba(0, 0, 0, 0.25) 0px 14px 28px, rgba(0, 0, 0, 0.22) 0px 10px 10px;
                            transform: scale(1.01);
                    }
                    img {
                            border-radius: 10px; 
                    }
                    #card1:hover{
                            box-shadow: rgba(0, 0, 0, 0.25) 0px 14px 28px, rgba(0, 0, 0, 0.22) 0px 10px 10px;
                            transform: scale(1.01);                           
                    }
                    #card2:hover{
                            box-shadow: rgba(0, 0, 0, 0.25) 0px 14px 28px, rgba(0, 0, 0, 0.22) 0px 10px 10px;
                            transform: scale(1.01);                           
                    }
            </style>
            </head>
            <body>
                <div id="container">
                    <div id="card1">
                        <h1>""" + text1 + """</h1>
                        <h3>""" + text2 + """</h3>
                        <h3>""" + text3 + """</h3>
                    </div>
                    <div id="card2">
                        <img src="https://img.freepik.com/free-vector/doctors-concept-illustration_114360-1515.jpg" alt="Doctor" width="200" height="135">
                        <h3>""" + text4 + """</h3>
                    </div>
                </div>
                <div>
                    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/""" + song_id + """?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
                </div>
            </body>
        </html>
        """
    return html

def get_song(creator, playlist_id):
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    temp = copy.copy(playlist)

    playlist_features = []

    for track in temp:
        # Get metadata
        playlist_features.append(track["track"]["id"])

    return random.choice(playlist_features)

def get_neutral():
    print("Executed")
    print(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"))
    text1 = "You are probably feeling Indifferent"
    text2 = "Prolonged cases of indifferance may show the signs of Schizoid personality disorder.It is a where a person shows very little, if any, interest and ability to form relationships with other people."
    text3 = "In a group setting, you can learn how to talk with others who are also learning and practicing new social skills. In time, group therapy may provide the support needed to make your social skills better."
    text4 = "With proper treatment and a skilled therapist, you can make a lot of progress and improve your quality of life."
    webview.create_window('Neutral', html=get_html(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"),text1,text2,text3,text4))
    webview.start()

def get_happiness():
    print("Executed")
    print(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"))
    text1 = "You are feeling Happy :)"
    text2 = "Happiness is a positive and pleasant emotion, ranging from contentment to intense joy."
    text3 = "Keep being Happy, and spread your positive energy to everyone around you!"
    text4 = "Moments of happiness may be triggered by positive life experiences or thoughts, but sometimes it may arise from no obvious cause."
    webview.create_window('Happiness', html=get_html(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"),text1,text2,text3,text4))
    webview.start()

def get_surprise():
    print("Executed")
    print(get_song("spotify","37i9dQZF1EVHGWrwldPRtj"))
    text1 = "You might br suffering from Bipolar Disorder"
    text2 = "Bipolar disorder is a mental illness that causes unusual shifts in a person’s mood, energy, activity levels, and concentration."
    text3 = "An effective treatment plan usually includes a combination of medication and psychotherapy, also called talk therapy."
    text4 = "Long-term, continuous treatment can help people manage these symptoms. Finding the treatment that works best for you may take trial and error."
    webview.create_window('Suprise', html=get_html(get_song("spotify","37i9dQZF1EVHGWrwldPRtj"),text1,text2,text3,text4))
    webview.start()

def get_sadness():
    print("Executed")
    print(get_song("spotify","37i9dQZF1EIeOoLWBf6eek"))
    text1 = "You might be suffering from Depression"
    text2 = "Depression is a common but serious mood disorder. It causes severe symptoms that affect how a person feels, thinks, and handles daily activities."
    text3 = "The earlier treatment begins, the more effective it is. Depression is usually treated with medication, psychotherapy, or a combination of the two."
    text4 = "No two people are affected the same way by depression. Finding the treatment that works best for you may take trial and error."
    webview.create_window('Sadness', html=get_html(get_song("spotify","37i9dQZF1EIeOoLWBf6eek"),text1,text2,text3,text4))
    webview.start()

def get_anger():
    print("Executed")
    print(get_song("spotify","37i9dQZF1EVHGWrwldPRtj"))
    text1 = "You might be suffering from Intermittent explosive disorder"
    text2 = "Intermittent explosive disorder is a mental health condition marked by frequent impulsive anger outbursts or aggression."
    text3 = "A licensed mental health professional — such as a psychiatrist, psychologist or clinical social worker — can diagnose your IED"
    text4 = "It's also very important to avoid alcohol and recreational drugs. These substances can increase the risk of violent behavior."
    webview.create_window('Anger', html=get_html(get_song("spotify","37i9dQZF1EVHGWrwldPRtj"),text1,text2,text3,text4))
    webview.start()

def get_disgust():
    print("Executed")
    print(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"))
    text1 = "You might be suffering from Obsessive-compulsive disorder"
    text2 = "Obsessive-compulsive disorder (OCD) is a long-lasting disorder in which a person experiences uncontrollable and recurring thoughts (obsessions), engages in repetitive behaviors (compulsions), or both."
    text3 = "Mental health professionals treat OCD with medications, psychotherapy, or a combination of treatments."
    text4 = "Although there is no cure for OCD, treatments help people manage their symptoms, engage in day-to-day activities, and lead full, active lives."
    webview.create_window('Disgust', html=get_html(get_song("spotify","37i9dQZF1DWVlYsZJXqdym"),text1,text2,text3,text4))
    webview.start()

def get_fear():
    print("Executed")
    print(get_song("spotify","6ACEZ7GO8z6PjIIf0jhlH8"))
    text1 = "You might suffering from Agoraphobia"
    text2 = "Agoraphobia is a type of anxiety disorder in which you fear and often avoid places or situations that might cause you to panic and make you feel trapped, helpless or embarrassed."
    text3 = "Lifestyle changes may help, including taking regular exercise, eating more healthily, and avoiding alcohol, drugs and drinks that contain caffeine, such as tea, coffee and cola."
    text4 = "Self-help techniques that can help during a panic attack include staying where you are, and slow, deep breathing."
    webview.create_window('Fear', html=get_html(get_song("spotify","6ACEZ7GO8z6PjIIf0jhlH8"),text1,text2,text3,text4))
    webview.start()


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    )
    return priors


def generate_priors(
        feature_map_list, shrinkage_list, image_size, min_boxes
):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(
        width,
        height,
        confidences,
        boxes,
        prob_threshold,
        iou_threshold=0.3,
        top_k=-1
):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1
        )
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4]
    )


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2,
         locations[..., :2] + locations[..., 2:] / 2],
        len(locations.shape) - 1
    )


def FER_live_cam():
    emotion_dict = {
        0: 'neutral',
        1: 'happiness',
        2: 'surprise',
        3: 'sadness',
        4: 'anger',
        5: 'disgust',
        6: 'fear'
    }

    # cap = cv2.VideoCapture('video1.mp4')
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('infer2-test.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    # Read ONNX model
    model = 'onnx_model.onnx'
    model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')

    # Read the Caffe face detector.
    model_path = 'RFB-320/RFB-320.caffemodel'
    proto_path = 'RFB-320/RFB-320.prototxt'
    net = dnn.readNetFromCaffe(proto_path, model_path)
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    last_save = time.time()
    emo_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_ori = frame
            # print("frame size: ", frame.shape)
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(rect, 1 / image_std, (width, height), 127))
            start_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = convert_locations_to_boxes(
                boxes, priors, center_variance, size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            boxes, labels, probs = predict(
                img_ori.shape[1],
                img_ori.shape[0],
                scores,
                boxes,
                threshold
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x1, y1, x2, y2) in boxes:
                w = x2 - x1
                h = y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)
                model.setInput(resize_frame)
                output = model.forward()
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                # print(f"FPS: {fps:.1f}")
                pred = emotion_dict[list(output[0]).index(max(output[0]))]
                cv2.rectangle(
                    img_ori,
                    (x1, y1),
                    (x2, y2),
                    (215, 5, 247),
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    pred,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (215, 5, 247),
                    2,
                    lineType=cv2.LINE_AA
                )

            result.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        current_time = time.time()
        if current_time - last_save >= 5:
            # Save emo_seq to file
            emo_str = '\n'.join(emo_seq)
            with open('emoseq.txt', 'w') as f:
                f.write(emo_str)
            emo_seq = []
            last_save = current_time

        # Add current emotion to sequence
        emo_seq.append(pred)
    cap.release()
    result.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()