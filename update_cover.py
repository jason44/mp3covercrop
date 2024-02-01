import cv2
import numpy as np
import os
import io
from PIL import Image
import argparse
import eyed3



def extract_cover(mp3_path, out):
    audiofile = eyed3.load(mp3_path)
    # Check if there is an existing album art
    if audiofile.tag.frame_set.get(b'APIC'):
        cover_frame = audiofile.tag.frame_set[b'APIC'][0]
        image_data = cover_frame.image_data
        image = Image.open(io.BytesIO(image_data))
        image.save(out)
        
        print(f"Cover image saved to: {out}")
    else:
        print("No album art found in the MP3 file.")


def crop_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    h, w, _ = image.shape
    cv2.line(gray, (w//4, 0), ((w//2)+(w//4), 0), (255, 255, 255), 1)
    cv2.line(gray, (w//4, h-1), ((w//2)+(w//4), h-1), (255, 255, 255), 1)

    gray = cv2.Canny(gray, 40, 80)
    # Canny edge detection results in very thin lines. Particularly, 
	# distinct lines aren't detected by findContours() so we dialate the lines 
    kernel = np.ones((2, 2),np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    cv2.imshow("Test", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # thresholding is unnecessary after edge detection 
    #_, binary_mask = cv2.threshold(gray, avg/2, 255, cv2.THRESH_BINARY)

    # Find largest contour in the bitmap
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # ignore crop if it's too small 
    if (w) < 300 or (h) < 300:
        print("BOX TOO SMALL RETURNING ORIGINAL IMAGE")
        return image 

    # send the cropped image in for a second pass (any more passes are generally redundant)
    cropped_gray = image[y:y+h, x:x+w]
    cropped_image = image[y:y+h, x:x+w]
    cropped_gray = cv2.cvtColor(cropped_gray, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.equalizeHist(cropped_gray)
    cropped_gray = cv2.Canny(cropped_gray, 50, 100)
    cropped_gray = cv2.morphologyEx(cropped_gray, cv2.MORPH_CLOSE, kernel, iterations = 1)

    contours, _ = cv2.findContours(cropped_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    if (w) < 300 or (h) < 300:
        print("BOX TOO SMALL RETURNING ORIGINAL IMAGE")
        return image 

    # 9 pixel border to compensate the dilation
    cropped_image = cropped_image[y:y+h, x+9:x+w-9]

    return cropped_image

def cleanup():
    if os.path.exists("cover.jpg"):
        os.remove("cover.jpg")
    if os.path.exists("cropped_image.jpg"):
        os.remove("cropped_image.jpg")


def embed_cover_art(mp3_path, cover_art_path):
    audiofile = eyed3.load(mp3_path)

    # Remove existing cover art, if any exist
    if audiofile.tag.frame_set.get(b'APIC'):
        del audiofile.tag.frame_set[b'APIC']

    try:
        with open(cover_art_path, 'rb') as cover_art_file:
            cover_art_data = cover_art_file.read()
    except FileNotFoundError:
        print(f"Cover art file '{cover_art_path}' not found.")
        return

    audiofile.tag.images.set(3, cover_art_data, 'image/jpeg')
    audiofile.tag.save()

    print(f"Cover art embedded successfully in '{mp3_path}'.")


def process_all():
    for file in os.listdir("./"):
        process_one(file)

def process_one(file):
        cleanup()
        _type = file.split('.')[-1]
        if _type == "mp3":
            # Load your image
            print(f"Working on: {file}")
            cover_path = 'cover.jpg'
            extract_cover(file, cover_path)
            image = cv2.imread(cover_path)
            cropped_image = crop_border(image)
            cv2.imwrite("cropped_image.jpg", cropped_image)
            embed_cover_art(file, "cropped_image.jpg")
            
parser = argparse.ArgumentParser(description='Crops cover images of mp3 files')
parser.add_argument('-i', '--input', type=str, help='mp3 file')
parser.add_argument('-a', '--all', action='store_true' , help='mp3 file')
parser.add_argument('-u', '--update', type=str, help='mp3 file')

args = parser.parse_args()
if args.input:
    process_one(args.input)
    exit()
if args.all:
    process_all
if args.update:
    embed_cover_art(args.update, "cropped_image.jpg")


