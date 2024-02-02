import cv2
import numpy as np
import os
import io
from PIL import Image
import argparse
import eyed3
import subprocess
import matplotlib.pyplot as plt

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

    gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
    gray = cv2.Canny(gray, 20, 50, apertureSize=3, L2gradient=True)

    height, width, _ = image.shape
    # forms a closed shape with vertical lines, to create a contour, 
    # length doesn't matter because we will square the image at the end 
    cv2.line(gray, (width//6, 4), (width-(width//6), 4), (255, 255, 255), 1)
    cv2.line(gray, (width//6, height-4), (width-(width//6), height-4), (255, 255, 255), 1)

    # Canny edge detection results in very thin lines. Particularly, 
	# distinct lines aren't detected by findContours() so we dialate the lines 
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
        if height * width < 510000 and not args.noupscale:
            return upscale(image)
        return image 

    # dilation was used so findContours can detect edges correctly,
    # so, the boundedRect will be offsetted proportional to the dilation, 
    # the solution is to square an image if it is already square-like
    cropped_image = image[y:y+h, x:x+w]
    height, width, _ = cropped_image.shape
    if float(width)/height < 1.35 and float(width)/height > 0.65:
        print("SQUARING")
        cropped_image = crop_to_square(cropped_image)
    else:
        # find "best" square
        cropped_gray = image[y:y+h, x:x+w]
        cropped_gray = cv2.cvtColor(cropped_gray, cv2.COLOR_BGR2GRAY)
        # the subject usually has more details(edges) than the background,
        # so for the second pass, use a high threshold so the view centers around the subject 
        cropped_gray = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        cropped_gray = cv2.Canny(cropped_gray, 135, 135, apertureSize=3, L2gradient=False)
        kernel1 = np.ones((2, 2),np.uint8)
        cropped_gray = cv2.morphologyEx(cropped_gray, cv2.MORPH_CLOSE, kernel1, iterations = 1)
        _, bitmask = cv2.threshold(cropped_gray, 10, 255, cv2.THRESH_BINARY)

        # iterate through candidates
        min_t = 99999999
        max_t = 0

        min_dim = min(width, height)
        max_x = width - min_dim
        max_y = height - min_dim
        center = (max_x//2, max_y//2)
        best = 0
        best_coord = (0, 0)

        pos_data, pixel_data, perim_data = [], [] , []
        for j in range(0, max_y+1):
            # modify jumps to speed up algorithm
            for i in range(0, max_x+1, 1):
                candidate = bitmask[j:j+min_dim, i:i+min_dim]
                dist = np.sqrt((center[0]-i)**2 + (center[1]-j)**2)
                square_sum = np.sum(candidate) 

                sq_contours, _ = cv2.findContours(candidate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                perim_sum = np.sum([cv2.arcLength(cnt, False) for cnt in sq_contours])*255

                pos_data.append(i)
                pixel_data.append(square_sum)
                perim_data.append(perim_sum)

                # dist*2440 penalizes the frame for moving away from the center with minimal gain
                square_sum -= 2440*dist

                if square_sum > best:
                    best = square_sum
                    best_coord = (i, j)
                if square_sum < min_t:
                    min_t = square_sum
                if square_sum > max_t:
                    max_t = square_sum

        # length perimeter is pretty much the same as just adding the pixel values
        plt.plot(pos_data, perim_data, color='blue', linewidth=3)
        plt.plot(pos_data, pixel_data, color='red', linewidth=3)
        plt.xlabel('x offset')
        plt.ylabel('perimeter value')
        plt.show()

        dist = np.sqrt((center[0]-best_coord[0])**2 + (center[1]-best_coord[1])**2) / (width*height*0.00005)
        print(f"distance: {dist}")
        print(f"range: {max_t-min_t}")
        # don't subtract dist when looking at this
        print(f"difference from center: {pixel_data[best_coord[0]-1]-pixel_data[center[0]-1]}")

        draw_bounds(bitmask, best_coord[0], best_coord[1], min_dim, min_dim)
        if dist < 1.9:
            best_coord = center
            draw_bounds(cropped_image, best_coord[0], best_coord[1], min_dim, min_dim)
        #draw_bounds(cropped_image, best_coord[0], best_coord[1], min_dim, min_dim)
        #draw_bounds(cropped_image, center[0], center[1], min_dim, min_dim)
        cropped_image = cropped_image[best_coord[1]:best_coord[1]+min_dim, best_coord[0]:best_coord[0]+min_dim]


    # use sr model for low quality images
    height, width, _ = cropped_image.shape
    if height * width < 520000 and not args.noupscale:
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
        height, width, _ = image.shape
        cv2.imwrite("tmp/lowres_crop.jpg", image)
        os.system(f"./realesrgan-ncnn-vulkan -i lowres_crop.jpg -o tmp/cropped_image.jpg -s 2")
        cropped_img = Image.open("tmp/cropped_image.jpg")
        # esrgan has really aggressive denoising, so we add some noise back into the upscaled image
        arr = np.array(cropped_img)
        avg = sum(cv2.mean(arr)) // 3.0
        noise = np.random.normal(0, (height*width/(200**2)), arr.shape) 
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
                cv2.imwrite("tmp/cropped_image.jpg", cropped_image)
            if not args.noembed:
                embed_cover_art(file, "tmp/cropped_image.jpg")
            
parser = argparse.ArgumentParser(description='Crops cover images of mp3 files')
parser.add_argument('-i', '--input', type=str, help='mp3 file')
parser.add_argument('-a', '--all', action='store_true' , help='mp3 file')
parser.add_argument('-u', '--update', type=str, help='mp3 file')
parser.add_argument('-d', '--directory', type=str, help='directory used by --all')
parser.add_argument('-nu', '--noupscale', action='store_true', help='do not use sr to upscale low res images')
parser.add_argument('-ne', '--noembed', action='store_true', help='do not embed output image into the input mp3')
parser.add_argument('-tmp', '--tmpfs', action='store_true', help='create tmpfs for storing images')

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
    tmp_path = os.path.join(os.chdir(), 'tmp')
    os.system(f'mount -t tmpfs -o size=2G tmpfs {tmp_path}') 
