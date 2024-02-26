import cv2
import numpy as np
import os
import io
from PIL import Image
import argparse
import eyed3
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import fftconvolve

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
    height, width, _ = image.shape
    if height == width:
        return image

    if args.force_center:
        cropped_image = crop_to_square(image)
        cropped_image = gradient_border(cropped_image)
        h, w, _ = cropped_image.shape
        if h * w < 445000 and not args.noupscale:
            print("UPSCALING")
            return upscale(cropped_image)
        else: return cropped_image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # thresholding is unnecessary after edge detection 
    #_, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.Canny(gray, 20, 50, apertureSize=3, L2gradient=True)

    # forms a closed shape with vertical lines, to create a contour, 
    startx = (width - height) // 2
    cv2.line(gray, (startx, 4), (startx+height, 4), (255, 255, 255), 2)
    cv2.line(gray, (startx, height-4), (startx+height, height-4), (255, 255, 255), 2)

    # Canny edge detection results in very thin lines. 
    # Particularly, distinct lines aren't detected by findContours() so dialate the lines 
    kernel = np.ones((5, 5),np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Find largest contour in the bitmap
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    draw_bounds(gray, x, y, w, h)

    # ignore crop if it's too small (usually the result of small foreground)
    if float(w) < float(width)/2.5 or float(h) < float(height)/2.5:
        print("BOX TOO SMALL. RETURNING ORIGINAL IMAGE")
        draw_bounds(gray, x, y, w, h)
        if height * width < 445000 and not args.noupscale:
            return upscale(image)
        return image 

    # square an image if it is already square-like
    cropped_image = image[y:y+h, x:x+w]
    height, width, _ = cropped_image.shape
    if float(width)/height < 1.35 and float(width)/height >= 0.65: 
        # check distance from the predicted square and clamp if it's small enough
        print("SQUARING")
        height, width, _ = image.shape
        min_dim = min(height, width)
        startx = (width - min_dim) // 2
        starty = (height - min_dim) // 2
        dist = np.sqrt((startx-x)**2 + (starty-y)**2)
        print(dist)
        if dist < 20.0:
            cropped_image = crop_to_square(image)
        else:
            cropped_image = crop_to_square(cropped_image)
        # draw gradient outline to 'solve' problem? (might be more of an asthetic change)
    else:
        # find "best" square
        cropped_gray = image[y:y+h, x:x+w]
        cropped_gray = cv2.cvtColor(cropped_gray, cv2.COLOR_BGR2GRAY)

        # the subject usually has more details(edges) and contrast than the background,
        # so for the second pass, use a higher threshold and gamma correction so reduce background edges
        gamma = 0.5
        cropped_gray = np.power(cropped_gray/float(np.max(cropped_gray)), gamma) * 255.0

        #cropped_gray = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        cropped_gray = cropped_gray.astype(np.uint8)
        cropped_gray = cv2.Canny(cropped_gray, 135, 135, apertureSize=3, L2gradient=False)
        kernel1 = np.ones((2, 2),np.uint8)
        cropped_gray = cv2.morphologyEx(cropped_gray, cv2.MORPH_CLOSE, kernel1, iterations = 1)
        _, bitmask = cv2.threshold(cropped_gray, 10, 255, cv2.THRESH_BINARY)

        # iterate through candidates
        min_dim = min(width, height)
        max_x = width - min_dim
        max_y = height - min_dim
        center = (max_x//2, max_y//2)
        best = 0
        best_coord = (0, 0)

        skip = 1
        pos_data, pixel_data = [], []
        for j in range(0, max_y+1):
            # modify jumps to speed up algorithm
            for i in range(0, max_x+1, skip):
                candidate = bitmask[j:j+min_dim, i:i+min_dim]
                dist = np.sqrt((center[0]-i)**2 + (center[1]-j)**2)
                square_sum = 255 * np.log(np.sum(candidate))**2

                pos_data.append(i)
                pixel_data.append(square_sum)

                if square_sum > best:
                    best = square_sum
                    best_coord = (i, j)

        # normalize pixel_data so the matplot grab becomes actually legible 
        pixel_data = minmax_normalize(np.array(pixel_data))
        pixel_data = con_smoothen(pixel_data, kernel_size=50)

        dydx = []
        for p1, p2 in zip(pixel_data, pixel_data[1:]):
            dydx.append((p2-p1)**2)
        # keep everything the same shape
        dydx.append(dydx[-1])
        dydx = minmax_normalize(np.array(dydx))
        #dydx = con_smoothen(dydx, kernel_size=1)

        adjvalues = pixel_data/dydx 
        max_raw = np.argmax(pixel_data)*skip
        max_adj = np.argmax(adjvalues)*skip
        best_coord = (max_adj, best_coord[1])

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
        axes[1, 0].imshow(cropped_image)
        axes[1, 1].imshow(bitmask)
        axes[1, 2].imshow(cropped_image[best_coord[1]:best_coord[1]+min_dim, best_coord[0]:best_coord[0]+min_dim])
        axes[0, 0].plot(pos_data, pixel_data, color='red', linewidth=3)
        axes[0, 0].set_xlabel('x offset')
        axes[0, 0].set_ylabel('perimeter value')
        axes[0, 1].plot(pos_data, dydx, color='blue', linewidth=3)
        axes[0, 1].set_xlabel('x offset')
        axes[0, 1].set_ylabel('derivative value')
        axes[0, 2].plot(pos_data, adjvalues, color='green', linewidth=3)
        axes[0, 2].set_xlabel('x offset')
        axes[0, 2].set_ylabel('final value')
        plt.tight_layout()
        #plt.show()

        dist = np.sqrt((center[0]-best_coord[0])**2 + (center[1]-best_coord[1])**2) / (width*height*0.00005)
        print(f"distance from center: {dist}")

        #draw_bounds(bitmask, center[0], center[1], min_dim, min_dim)
        #draw_bounds(bitmask, best_coord[0], best_coord[1], min_dim, min_dim)
        if dist < 1.8:
            best_coord = center
            #draw_bounds(cropped_image, best_coord[0], best_coord[1], min_dim, min_dim)
        #draw_bounds(cropped_image, best_coord[0], best_coord[1], min_dim, min_dim)
        #draw_bounds(cropped_image, center[0], center[1], min_dim, min_dim)
        cropped_image = cropped_image[best_coord[1]:best_coord[1]+min_dim, best_coord[0]:best_coord[0]+min_dim]

    cropped_image = gradient_border(cropped_image)

    # use sr model for low quality images
    height, width, _ = cropped_image.shape
    if height * width < 445000 and not args.noupscale:
        print("UPSCALING")
        cropped_image = upscale(cropped_image)
    
    return cropped_image

def con_smoothen(data, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')

def minmax_normalize(data):
    min = np.min(data)
    max = np.max(data)
    # prevent zero division error by excluding 0
    return 0.001 + (data - min)*0.999 / (max - min)

def mean_normalize(data):
    min = np.min(data)
    max = np.max(data)
    mean = np.mean(data)
    return np.abs(data - mean) / (max - min)

def gradient_border(image, border_size=6):
    blurred = cv2.stackBlur(image, (227, 227))
    topedge = blurred[:border_size, :]
    botedge = blurred[:-border_size:-1, :] 
    leftedge = blurred[:, :border_size] 
    rightedge = blurred[:, :-border_size:-1]
    edgeavg = (np.average(topedge) + np.average(botedge) + np.average(leftedge) + np.average(rightedge)) / 4
    blurred = np.clip(blurred*1.75, 0, min(edgeavg+20, 255))
    image[:border_size, :] = topedge
    image[:-border_size:-1, :] = botedge
    image[:, :border_size] = leftedge
    image[:, :-border_size:-1] = rightedge
    return image

def crop_to_square(image):
    height, width, _ = image.shape
    min_dim = min(height, width)
    starty = (height - min_dim) // 2
    endy = starty + min_dim
    # since all images are 16:9, we always have starty = 0
    startx = (width - min_dim) // 2
    endx = startx + min_dim

    cropped_image = image[starty:endy, startx:endx]

    return cropped_image

def draw_bounds(image, x, y, w, h):
    cv2.line(image, (x, y), (x+w, y), (255, 0, 0), 2)
    cv2.line(image, (x, y), (x, y+h), (255, 0, 0), 2)
    cv2.line(image, (x, y+h), (x+w, y+h), (255, 0, 0), 2)
    cv2.line(image, (x+w, y), (x+w, y+h), (255, 0, 0), 2)

    #cv2.imshow("N", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def upscale(image):
        height, width, _ = image.shape
        cv2.imwrite("tmp/lowres_crop.jpg", image)
        os.system(f"./realesrgan-ncnn-vulkan -i tmp/lowres_crop.jpg -o tmp/cropped_image.jpg -s 2")
        cropped_img = Image.open("tmp/cropped_image.jpg")
        # esrgan has really aggressive denoising, so we add some noise back into the upscaled image
        arr = np.array(cropped_img)
        avg = sum(cv2.mean(arr)) // 3.0
        noise = np.random.normal(0, (height*width/(300**2)), arr.shape) 
        noisy = np.clip(arr + noise, 0, 255).astype('uint8')
        out = Image.fromarray(noisy)
        out.save("tmp/cropped_image.jpg", 'JPEG', quality=75)
        return None

def cleanup():
    if os.path.exists("tmp/cover.jpg"):
        os.remove("tmp/cover.jpg")
    if os.path.exists("tmp/cropped_image.jpg"):
        os.remove("tmp/cropped_image.jpg")
    if os.path.exists("tmp/lowres_crop.jpg"):
        os.remove("tmp/lowres_crop.jpg")


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


def process_all(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        process_one(path)

def process_one(file):
        cleanup()
        _type = file.split('.')[-1]
        if _type == "mp3":
            # Load your image
            print(f"Working on: {file}")
            cover_path = 'tmp/cover.jpg'
            extract_cover(file, cover_path)
            image = cv2.imread(cover_path)
            cropped_image = crop_border(image)
            if cropped_image is not None:
                # draw gradient border
                cv2.imwrite("tmp/cropped_image.jpg", cropped_image)
            if not args.noembed:
                embed_cover_art(file, "tmp/cropped_image.jpg")
            
parser = argparse.ArgumentParser(description='Crops cover images of mp3 files')
parser.add_argument('-i', '--input', type=str, help='mp3 file')
parser.add_argument('-a', '--all', action='store_true' , help='mp3 file')
parser.add_argument('-u', '--update', type=str, help='update mp3 file with cropped_image.jpg')
parser.add_argument('-d', '--directory', type=str, help='directory used by --all')
parser.add_argument('-nu', '--noupscale', action='store_true', help='do not use sr to upscale low res images')
parser.add_argument('-ne', '--noembed', action='store_true', help='do not embed output image into the input mp3')
parser.add_argument('-tmp', '--tmpfs', action='store_true', help='create tmpfs for storing images')
parser.add_argument('-fc', '--force-center', action='store_true', help='force the frame center to align with the horizontal center')

args = parser.parse_args()
if args.input:
    process_one(args.input)
    exit()
if args.all:
    if args.directory:
        process_all(args.directory)
    else:
        process_all('./')
if args.update:
    embed_cover_art(args.update, "tmp/cropped_image.jpg")
if not os.path.exists('tmp'):
    os.mkdir('tmp')
if args.tmpfs:
    tmp_path = os.path.join(os.curdir, 'tmp')
    os.system(f'sudo mount -t tmpfs -o size=2G tmpfs {tmp_path}') 
