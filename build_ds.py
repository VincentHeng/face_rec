from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os 
import csv

def apply_op(dirname, filter):
	iname_list = [] 
	for iname in os.listdir(dirname):
		iname_list.append(iname)
	nTag = ""
	tag = ""
	if filter == ImageOps.mirror:
		nTag = "_nmirror"
		tag = "_mirror"
	if filter == ImageOps.invert:
		nTag = "_ninvert"
		tag = "_invert"
	if filter == ImageOps.grayscale:
		nTag = "_ngs"
		tag = "_gs"
	if filter == ImageOps.equalize:
		nTag = "_neqz"
		tag = "_eqz"
	for iname in iname_list:
		# updated all the old names
		im = Image.open("{}/{}".format(dirname, iname))
		new_im = filter(im)
		ext_index = iname.find(".")
		im.close()
		old_name = "{}/{}".format(dirname, iname)
		updated_name = "{}/{}".format(dirname, iname[:ext_index] + nTag + iname[ext_index:])
		os.rename(old_name, updated_name)
		new_im.save("{}/{}".format(dirname, iname[:ext_index] + tag + iname[ext_index:]))

def apply_blur(dirname):
	iname_list = [] 
	for iname in os.listdir(dirname):
		iname_list.append(iname)
	for iname in iname_list:
		# updated all the old names
		im = Image.open("{}/{}".format(dirname, iname))
		new_im = im.filter(ImageFilter.BLUR)
		ext_index = iname.find(".")
		im.close()
		old_name = "{}/{}".format(dirname, iname)
		updated_name = "{}/{}".format(dirname, iname[:ext_index] + "_nblur" + iname[ext_index:])
		os.rename(old_name, updated_name)
		new_im.save("{}/{}".format(dirname, iname[:ext_index] + "_blur" + iname[ext_index:]))

def adj_br(dirname):
	iname_list = [] 
	for iname in os.listdir(dirname):
		iname_list.append(iname)
	for iname in iname_list:
		# updated all the old names
		im = Image.open("{}/{}".format(dirname, iname))
		enhancer = ImageEnhance.Brightness(im)

		new_im_brup = enhancer.enhance(1.2)
		new_im_brdown = enhancer.enhance(0.8)

		# new_im = im.filter(ImageFilter.BLUR)
		ext_index = iname.find(".")
		im.close()
		old_name = "{}/{}".format(dirname, iname)
		updated_name = "{}/{}".format(dirname, iname[:ext_index] + "_nAdjBr" + iname[ext_index:])
		os.rename(old_name, updated_name)

		new_im_brup.save("{}/{}".format(dirname, iname[:ext_index] + "_incBr" + iname[ext_index:]))
		new_im_brdown.save("{}/{}".format(dirname, iname[:ext_index] + "_decBr" + iname[ext_index:]))


def mod_photos(dirname):
	#apply_op(dirname, ImageOps.mirror) #1, doubles, MIRROR
	# apply_op(dirname, ImageOps.invert) #2, doubles, INVERT
	# apply_op(dirname, ImageOps.grayscale) #3, doubles, GRAYSCALE
	apply_op(dirname, ImageOps.equalize) #4, doubles, equalize
	apply_blur(dirname) #5 doubles
	adj_br(dirname) #6 triples


def rename_photos(dirname, name):
	name_set = {} # dictionary of name tallies
	iname_queue = []
	count = 0
	for iname in os.listdir(dirname):
		iname_queue.append(iname)
		count += 1

	# print(name_set)
	# print(iname_queue)

	for iname in iname_queue:
		os.rename('{}/{}'.format(dirname, iname), "{}/{}{}.jpg".format(dirname, name, count))
		count -= 1

def create_csv(fname, dirname):
	mp = {'vincent': 0, 'grandma': 1, 'grandpa': 2, 'mom': 3}

	with open(fname, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		for folder in os.listdir(dirname):
			# print(folder)
			for iname in os.listdir('{}/{}'.format(dirname, folder)):
				print(iname)
				image_path = '{}/{}'.format(folder, iname)
				csv_writer.writerow([image_path, mp[iname2name(iname)]])

def iname2name(iname):
	idx = None
	for i, c in enumerate(iname):
		if c.isdigit():
			idx = i
			break
	return iname[0: idx]


def main():
	# first loop resolution to 128x128	
	# name = 'edwardcpy'
	# dirname = 'dataset/{}'.format(name)
	for name in ['vincent', 'grandma', 'grandpa', 'mom']:
		dirname = 'infer_set/{}'.format(name)
		for i in os.listdir(dirname):
			im = Image.open("%s/%s" % (dirname, i))
			im = im.resize((128, 128))
			# im = ImageOps.grayscale(im)
			im.save("%s/%s" % (dirname, i))

		
		mod_photos(dirname)
		rename_photos(dirname, name)

	csv_name = 'infer_set.csv'
	ifolder = 'infer_set'
	create_csv(csv_name, ifolder)
	
if __name__ == '__main__':
	main()