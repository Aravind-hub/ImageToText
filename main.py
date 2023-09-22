from PIL import Image
import pytesseract
import nltk
from textblob import TextBlob
import cv2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    img_file = Image.open('handwritten.JPG')

    #image processing
    image = cv2.imread('handwritten.JPGx')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('threshold image', threshold_img)
    cv2.waitKey(0)

    text = pytesseract.image_to_string(img_file, lang='eng')
    print(text)
    with open('output.txt', 'w') as f:
        f.write(text)

    blob = TextBlob(text)
    sentiment = blob.sentiment
    print(sentiment.polarity)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
