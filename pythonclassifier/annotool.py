import SmartAnnotator as Sa
from bottle import request, route, run, template
import json

@route('/', method='POST')
def train():

# if __name__ == '__main__':

	
	data = request.json
	# data = '["foo", {"bar":["baz", null, 1.0, 2]}]'

	# intitiate SmartAnnotator
	sa = Sa.SmartAnnotator("uploaded_images", 10, 2)

	print data[0]
	# add positive sample points
	for i in range (0, len(data[0]["pos"])):
		x_point = data[0]["pos"][i][1:4]
		# print x_point,","
		y_point = data[0]["pos"][i][6:9]
		# print y_point
		sa.add_positive_sample(int(x_point),int(y_point))

	print "___"
	print data[1]


	# add negative sample points
	print len(data[1]["neg"])
	for j in range (0, len(data[1]["neg"])):
		x_point = data[1]["neg"][j][1:4]
		# print x_point,","
		y_point = data[1]["neg"][j][6:9]
		# print y_point
		sa.add_negative_sample(int(x_point),int(y_point))


	print "TESTTTT"
	sa.train_command()
	sa.test_command()

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
		detections.append([x,y,p])

	print "SECONDS"

	return json.dumps(detections)


run(host='localhost', port=8080, debug=True)
