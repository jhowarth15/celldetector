from bottle import request, route, run, template
import json

@route('/', method='POST')
def train():

	data = request.json
	# data = '["foo", {"bar":["baz", null, 1.0, 2]}]'

	print data[0]["foo"]

	return 5*data[0]["foo"]

run(host='localhost', port=8080, debug=True)
