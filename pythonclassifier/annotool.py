import SmartAnnotator as Sa
from bottle import request, route, run, template
import json
import os, os.path, shutil

@route('/', method='POST')
def train():

# if __name__ == '__main__':

	
	data = request.json
	# data = '["foo", {"bar":["baz", null, 1.0, 2]}]'

	# number of channels
	nChannels = data[2]["settings"][0]
	print "No Channels: ", nChannels

	# number of frames
	frameCount = data[2]["settings"][2]
	print "No Frames: ", frameCount

	classifierSet = data[2]["settings"][3]
	print "Classifier Set: ", classifierSet

	# intitiate SmartAnnotator
	try:
		sa
		del sa
		print "RETRAIN"
	except NameError:
		print 'FIRST TRAINING'

	global sa
	sa = Sa.SmartAnnotator("uploaded_images", frameCount, nChannels, classifierSet)

	print data[0]
	# add positive sample points
	for i in range (0, len(data[0]["pos"])):
		x_point = data[0]["pos"][i][1:4]
		# print x_point,","
		y_point = data[0]["pos"][i][6:9]
		# print y_point
		point_frame = data[0]["pos"][i][11:15]
		print point_frame
		sa.add_positive_sample(int(x_point),int(y_point),int(point_frame))

	print "_"*10
	print data[1]


	# add negative sample points
	print len(data[1]["neg"])
	for j in range (0, len(data[1]["neg"])):
		x_point = data[1]["neg"][j][1:4]
		# print x_point,","
		y_point = data[1]["neg"][j][6:9]
		# print y_point
		point_frame = data[0]["pos"][i][11:15]
		print point_frame
		sa.add_negative_sample(int(x_point),int(y_point),int(point_frame))

	sa.train_command()

	return


@route('/test', method='POST')
def test():

	data = request.json

	# frame in question
	nframe = int(data)
	print "Frame: ", nframe

	# sa.test_command(nframe)

	sa._test_n_frame(nframe)

	detections = []


	count = 0
	# iterate over the dots
	for dot in sa.dots:
        # display only dots with probability larger than threshold
		# if dot.probability < sa.slider:
		# 	continue

		x, y, p = dot.x, dot.y, int(dot.probability*100)
		count = count+1
		print "Cell ", count, ": ", "[", x, ", ", y, "], P: ", dot.probability
		detections.append([x,y,p,nframe])

	# delete all the pngs
	# print "DELETING SAVED PNGS.."
	# folder = '/Users/joshuahowarth/dev/celldetector/pythonclassifier/uploaded_images/'
	# for the_file in os.listdir(folder):
	# 	file_path = os.path.join(folder, the_file)
	# 	try:
	# 		if os.path.isfile(file_path) and the_file.endswith(".png"):
	# 			os.unlink(file_path)
	# 			#elif os.path.isdir(file_path): shutil.rmtree(file_path)
	# 	except Exception as e:
	# 		print(e)	
	# print 'DONE.'

	# return the detections
	return json.dumps(detections)


run(host='localhost', port=8080, debug=True)
