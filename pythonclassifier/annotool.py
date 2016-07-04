import SmartAnnotator as Sa
from bottle import request, route, run, template
import json

@route('/', method='POST')
def train():

# if __name__ == '__main__':

	data = request.json
	# data = '["foo", {"bar":["baz", null, 1.0, 2]}]'

	print data[0]["pos"]

	sa = Sa.SmartAnnotator("png_rgb", 10, 2)

	print "TESTTTT"
	sa.train_command()
	sa.test_command()

	detections = []

	count = 0
	# iterate over the dots
	for dot in sa.dots:
        # display only dots with probability larger than threshold
		if dot.probability < sa.slider:
			continue

		x, y = dot.x, dot.y
		count = count+1
		print "Cell ", count, ": ", "[", x, ", ", y, "]. "
		detections.append([x,y])

	print "SECONDS"

	return json.dumps(detections)


run(host='localhost', port=8080, debug=True)
