import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
import argparse
from autocorrect import Speller

spell = Speller(lang='en')

def checkAndCorrect(word):
  return spell(word)

def removeNoice(image, kernel_size=3):
  return cv2.medianBlur(image, 3)


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--input", help="Input Image File Path", required=True)
  parser.add_argument("--minconf", help="Minimun confidence For OCR", default=0, type=int)
  parser.add_argument("--removenoice", help="Remove Noice in image by applying median blur filter. Use this Filter if the image size is large. It is not recomanded for small sized Image.", default=False, type=bool)
  parser.add_argument("--kernelsize", help="Kernel size for applying median blure filter.", default=3, type=int)
  parser.add_argument("--spellconf", help="confidence below which the spell check and correction should be performed.", default=90, type=int)
  args = parser.parse_args()

  filepath = args.input
  min_conf = args.minconf
  remove_noice = args.removenoice
  kernel_size = args.kernelsize
  spell_conf = args.spellconf

  if not os.path.exists(filepath):
    print("input file does not exists:", filepath)
    sys.exit(1)

  image = cv2.imread(filepath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  print(image.shape)
  if remove_noice:
    image = removeNoice(image, kernel_size=kernel_size)

  data = pytesseract.image_to_data(image, lang = 'eng', output_type='data.frame',  config='--oem 1')

  data = data[data.conf >  min_conf]
  data.dropna(inplace=True)
  
  for idx in data.index:
    print(data['text'][idx])
    if data['conf'][idx]<spell_conf:
      data['text'][idx] = checkAndCorrect(data['text'][idx])

  lines = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(list(x))).tolist()
  print(lines)




if __name__ == "__main__":
  main()
# image = cv2.imread('Data\Assignment_1.tiff')
# image = cv2.cvtColor(cv2.medianBlur(image,3), cv2.COLOR_BGR2RGB)
# text = pytesseract.image_to_data(image, lang = 'eng', output_type='data.frame',  config='--oem 1')

# text = text[text.conf <20]

# # text['new'] = text['text']*
# lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(list(x))).tolist()


# results = text

# text['corrected_text'] = text['text'].apply(spell)

# err = text[text['new']!=text['text']].reset_index(drop=True)

# for i in range(0, len(err["text"])):
      
#     # We can then extract the bounding box coordinates
#     # of the text region from  the current result
#     x = err["left"][i]
#     y = err["top"][i]
#     w = err["width"][i]
#     h = err["height"][i]
#     text_ = err['text'][i]
#     # We will also extract the OCR text itself along
#     # with the confidence of the text localization
#     _text = err["new"][i]
#     conf = int(err["conf"][i])
      
#     # filter out weak confidence text localizations
#     if conf <90:
          
#         # We will display the confidence and text to
#         # our terminal
#         # print("Confidence: {}".format(conf))
#         # print("Text: {}".format(text))
#         # print("")
          
#         # We then strip out non-ASCII text so we can
#         # draw the text on the image We will be using
#         # OpenCV, then draw a bounding box around the
#         # text along with the text itself
#         cv2.rectangle(image,
#                       (x, y),
#                       (x + w, y + h),
#                       (255, 255, 0), 2)
#         cv2.putText(image,
#                     text_,
#                     (x, y -3), 
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (255, 0, 0), 2)
#         cv2.putText(image,
#                     _text+' '+str(conf),
#                     (x, y +h +10), 
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (255, 0, 255), 2)
        
# cv2.imwrite("Image.png",img=image)