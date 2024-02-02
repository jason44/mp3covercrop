import cv2
import numpy as np
import os
import io
from PIL import Image
import argparse
import eyed3
import subprocess


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

    # thresholding is unnecessary after edge detection 
    #_, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)

    height, width, _ = image.shape
    cv2.line(gray, (width//6, 4), (width-(width//6), 4), (255, 255, 255), 3)
    cv2.line(gray, (width//6, height-4), (width-(width//6), height-4), (255, 255, 255), 3)

    gray = cv2.Canny(gray, 40, 100, apertureSize=7, L2gradient=True)

    # Canny edge detection results in very thin lines. Particularly, 
	# distinct lines aren't detected by findContours() so we dialate the lines 
    kernel = np.ones((2, 2),np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Find largest contour in the bitmap
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    #draw_bounds(gray, x, y, w, h)

    # ignore crop if it's too small 
    if float(w) < float(width)/2.2 or float(h) < float(height)/2.2:
        print("BOX TOO SMALL. RETURNING ORIGINAL IMAGE")
        draw_bounds(gray, x, y, w, h)
        if height * width < 679600:
            return upscale(image)
        return image 

    # unnecessary since we can just square images that are already square-like 
    cropped_gray = image[y:y+h, x:x+w]
    cropped_image = image[y:y+h, x:x+w]
    #_, cropped_gray = cv2.threshold(cropped_gray, 5, 255, cv2.THRESH_BINARY)
    cropped_gray = cv2.cvtColor(cropped_gray, cv2.COLOR_BGR2GRAY)
    #cropped_gray = cv2.equalizeHist(cropped_gray)
    cropped_gray = cv2.Canny(cropped_gray, 100, 40, apertureSize=3)
    kernel = np.ones((2, 2),np.uint8)
    cropped_gray = cv2.morphologyEx(cropped_gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    contours, _ = cv2.findContours(cropped_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    #draw_bounds(cropped_gray, x, y, w, h)

    if float(w) < float(width)/2.2 or float(h) < float(height)/2.2:
        print("BOX TOO SMALL. RETURNING ORIGINAL IMAGE (SECOND PASS).\n\
              FALLING BACK TO FIRST PASS IMAGE")
        #draw_bounds(cropped_gray, x, y, w, h)
    else:
        cropped_image = cropped_image[y:y+h, x:x+w]

    # dilation was used so findContours can detect edges correctly,
    # so, the boundedRect will be offsetted proportional to the dilation, 
    # the solution is to square an image if it is already square-like
    height, width, _ = cropped_image.shape
    # this is always true lol
    if float(width)/height < 1.15 or float(width)/height > 0.85:
        print("SQUARING")
        cropped_image = crop_to_square(cropped_image)

    # use sr model for low quality images
    height, width, _ = cropped_image.shape
    if height * width < 679600:
        print("UPSCALING")
        return upscale(cropped_image)

    return cropped_image

def crop_to_square(image):
    height, width, _ = image.shape
    min_dim = min(height, width)
    startx = (height - min_dim) // 2
    endx = startx + min_dim
    # since all images are 16:9, we always have starty = 0
    starty = (width - min_dim) // 2
    endy = starty + min_dim

    cropped_image = image[startx:endx, starty:endy]

    return cropped_image

def draw_bounds(image, x, y, w, h):
    cv2.line(image, (x, y), (x+w, y), (255, 0, 0), 2)
    cv2.line(image, (x, y), (x, y+h), (255, 0, 0), 2)
    cv2.line(image, (x, y+h), (x+w, y+h), (255, 0, 0), 2)
    cv2.line(image, (x+w, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("N", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def upscale(image):
        # apply some noise before upscaling
        height, width, _ = image.shape
        cv2.imwrite("lowres_crop.jpg", image)
        os.system(f"./realesrgan-ncnn-vulkan -i lowres_crop.jpg -o cropped_image.jpg -s 2")
        cropped_img = Image.open("cropped_image.jpg")
        # esrgan has really aggressive denoising, so we add some noise back into the upscaled image
        arr = np.array(cropped_img)
        avg = sum(cv2.mean(arr)) // 3.0
        noise = np.random.normal(0, (height*width/(215**2)), arr.shape) 
        noisy = np.clip(arr + noise, 0, 255).astype('uint8')
        print(noise)
        out = Image.fromarray(noisy)
        out.save("cropped_image.jpg", 'JPEG', quality=75)
        return None

def cleanup():
    if os.path.exists("cover.jpg"):
        os.remove("cover.jpg")
    if os.path.exists("cropped_image.jpg"):
        os.remove("cropped_image.jpg")
    if os.path.exists("lowres_crop.jpg"):
        os.remove("lowres_crop.jpg")


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
            if cropped_image is not None:
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
    process_all()
if args.update:
    embed_cover_art(args.update, "cropped_image.jpg")


