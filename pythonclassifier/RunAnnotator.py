import SmartAnnotator as Sa

if __name__ == '__main__':

	sa = Sa.SmartAnnotator("png_rgb", 10, 2)

	print "TESTTTT"
	sa.train_command()
	sa.test_command()

	count = 0
	# iterate over the dots
	for dot in sa.dots:
        # display only dots with probability larger than threshold
		if dot.probability < sa.slider:
			continue

		x, y = dot.x, dot.y
		count = count+1
		print "Cell ", count, ": ", "[", x, ", ", y, "]. "

print "SECONDS"