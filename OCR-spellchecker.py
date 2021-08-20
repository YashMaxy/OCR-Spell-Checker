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

def removeNoise(image, kernel_size=3):
  return cv2.medianBlur(image, 3)


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--input", help="Input Image File Path", required=True)
  parser.add_argument("--minconf", help="Minimun confidence For OCR", default=0, type=int)
  parser.add_argument("--removenoise", help="Remove Noise in image by applying median blur filter. Use this Filter if the image size is large. It is not recomanded for small sized Image.", default=False, type=bool)
  parser.add_argument("--kernelsize", help="Kernel size for applying median blure filter.", default=3, type=int)
  parser.add_argument("--spellconf", help="confidence below which the spell check and correction should be performed.", default=90, type=int)
  args = parser.parse_args()

  filepath = args.input
  min_conf = args.minconf
  remove_noise = args.removenoise
  kernel_size = args.kernelsize
  spell_conf = args.spellconf

  if not os.path.exists(filepath):
    print("input file does not exists:", filepath)
    sys.exit(1)

  image = cv2.imread(filepath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  print(image.shape)
  
  if remove_noise:
    image = removeNoise(image, kernel_size=kernel_size)

  data = pytesseract.image_to_data(image, lang = 'eng', output_type='data.frame',  config='--oem 1')
  
#   removing Detection which hase confidence value less than min_conf value. 
  data = data[data.conf >  min_conf]
  data.dropna(inplace=True)
  
#   Check and correct the spelling whose confidence value is less than spell_conf.
  for idx in data.index:
    if data['conf'][idx]<spell_conf:
      data['text'][idx] = checkAndCorrect(data['text'][idx])

#   generating list of string from Dataframe and save the text in text.txt file
  lines = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(list(x))).tolist()
  print(lines)
  textfile = open("text.txt", "w")
  for element in lines:
      textfile.write(element + "\n")
  textfile.close()



if __name__ == "__main__":
  main()
