from PIL import Image
import os, sys

path = "asl_alphabet_test/"
dirs1 = os.listdir( path )

for item1 in dirs1:
	dirs = os.listdir(path+item1)
	for item in dirs:
		im = Image.open(path+item1+'/'+item)
		if im:
			f, e = os.path.splitext(path+item1+'/'+item)
			imResize = im.resize((200,200), Image.ANTIALIAS)
			imResize.save(f+'.jpg', 'JPEG', quality=90)