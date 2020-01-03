
''' Wrote by Jason Huang, 2019.07.25 '''
import sys
import os
import xmltodict
import json

# print("dict : ", sys.argv[1])
# argv = "F:/KunShanVXN/20190618/data/test"
def xml2json(argv):
	print("dict : ", argv)
	fileNumber = 0
	for dirpath, dirnames, filenames in os.walk(argv):
		for file in filenames:
			fullPath = os.path.join(dirpath, file)
			tmp = os.path.splitext(fullPath)
			prefix = tmp[0]
			postfix = tmp[1]
			if postfix == '.xml':
				xml_file = open(fullPath, 'r', encoding='utf-8')
				xml_str = xml_file.read()
				json_str = xmltodict.parse(xml_str)
				f = open(prefix + '.json', 'w')
				f.write(json.dumps(json_str))
				f.close()
				xml_file.close()
				fileNumber += 1
	print("Convert xml2json finished, total file number: ", fileNumber)
