from bottle import route, run, template

@route('/')
def nodata():
    return "No data provided."

@route('/<data>')
def train(data=90210):
    return 2*data

run(host='localhost', port=8080, debug=True)
