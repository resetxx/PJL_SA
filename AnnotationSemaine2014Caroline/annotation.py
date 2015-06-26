import xml.etree.ElementTree as ET
from glob import glob
import os.path as op

def generateInfo_schema(filename): # get polarity info from schema
	aaTree = ET.parse(filename)
	aaRoot = aaTree.getroot()

	units = aaRoot.getiterator("unit")
	schemaInfos = [] # schema id, start and end index of appraisalItem unit, polarity


	for schema in aaRoot.findall("./schema/characterisation[type='userAttitude'].."): # node"schema"
		info = []
		unitIds = []
		if(schema.find(".//embedded-unit") is not None): #schema having childnode"embedded-unit"
			info.append(schema.get("id")) #schema id

			for embeddedUnit in schema.findall(".//embedded-unit"):
				unitIds.append(embeddedUnit.get("id"))
			
			for unit in units:#find unit node which type is 'appraisalItem'
				# store start/end index of appraisalItem unit
				for i in xrange(len(unitIds)):
					if (unit.attrib["id"]==unitIds[i]) and (unit.find(".//type").text=='appraisalItem'):
						start = unit.find(".//start/singlePosition").get("index")
						end = unit.find(".//end/singlePosition").get("index")
						info.append(int(start))
						info.append(int(end))

			info.append(schema.find(".//feature[@name='polarity']").text) #polarity

			schemaInfos.append(info)

	schemaInfos = sorted(schemaInfos, key = lambda l:l[0])
	return schemaInfos

def generateInfo_unit(filename): # get polarity info from unit
	aaTree = ET.parse(filename)
	aaRoot = aaTree.getroot()
	unitInfos = []

	for unit in aaRoot.findall("./unit/characterisation[type='operatorUtterance'].."): 
		info = []
		polarity = unit.find(".//feature[@name='attitudePolarity']").text

		if(polarity == 'positive') or (polarity == 'negative'): 
		# not consider neutral/undefined attitude
			info.append(unit.get("id")) #unit id			
			start = unit.find(".//start/singlePosition").get("index")
			end = unit.find(".//end/singlePosition").get("index")
			info.append(int(start))
			info.append(int(end))
			info.append(polarity)

			unitInfos.append(info)

	for unit in aaRoot.findall("./unit/characterisation[type='nonVerbal'].."): 
		info = []
		polarity = unit.find(".//feature[@name='polarityAttitude']").text

		if(polarity == 'positive') or (polarity == 'negative'): 
			info.append(unit.get("id"))
			start = unit.find(".//start/singlePosition").get("index")
			end = unit.find(".//end/singlePosition").get("index")
			info.append(int(start))
			info.append(int(end))
			info.append(polarity)

			unitInfos.append(info)

	unitInfos = sorted(unitInfos, key = lambda l:l[0]) # sort the list by start position index
	return unitInfos

# assign polarity to each paragraph(turn of conversation)
def setPolByLine(file_aa, file_ac, schemaInfo, unitInfo):
	textsPos = []
	textsNeg = []
	textsNeu = []
	
	aaTree = ET.parse(file_aa)
	aaRoot = aaTree.getroot()
	
	with open(file_ac, "r") as myfile: # replace tab \t with espace
		text = myfile.read().replace('	',' ')

	for unitPara in aaRoot.findall("./unit/characterisation[type='paragraph'].."):
		start = int(unitPara.find(".//start/singlePosition").get("index"))
		end = int(unitPara.find(".//end/singlePosition").get("index"))
		flag = 1
		for sch in schemaInfo:
			if start <= sch[1] <= end and sch[3] == 'positive' and flag == 1:
				# remove unwanted text like'3310 recording05_session025_Spike_operator'
				i = text[start:end].find('user') 
				j = text[start:end].find('operator')
				if  i != -1:
					textsPos.append(text[start+i+5:end]) # length of 'user ':5
				elif j != -1:
					textsPos.append(text[start+j+9:end]) # length of 'operator ':9
				flag = 0
			elif start <= sch[1] <= end and sch[3] == 'negative' and flag == 1:
				i = text[start:end].find('user')
				j = text[start:end].find('operator')
				if  i != -1:
					textsNeg.append(text[start+i+5:end])
				elif j != -1:
					textsNeg.append(text[start+j+9:end])
				flag = 0

		for unit in unitInfo:
			if start <= unit[1] <= end and unit[3] == 'positive' and flag == 1:
				i = text[start:end].find('user')
				j = text[start:end].find('operator')
				if  i != -1:
					textsPos.append(text[start+i+5:end])
				elif j != -1:
					textsPos.append(text[start+j+9:end])
				flag = 0
			elif start <= unit[1] <= end and unit[3] == 'negative' and flag == 1:
				i = text[start:end].find('user')
				j = text[start:end].find('operator')
				if  i != -1:
					textsNeg.append(text[start+i+5:end])
				elif j != -1:
					textsNeg.append(text[start+j+9:end])
				flag = 0
				
		if flag == 1:
			i = text[start:end].find('user')
			j = text[start:end].find('operator')
			if  i != -1:
				textsNeu.append(text[start+i+5:end])
			elif j != -1:
				textsNeu.append(text[start+j+9:end])

	return textsPos, textsNeg, textsNeu

# verify if one paragraph has more than one polarity
def existConflits():
	filenames_aa = sorted(glob(op.join('.', 'aa1', '*.aa')))
	filenames_ac = sorted(glob(op.join('.', 'ac1', '*.ac')))

	unitInfos = []
	schemaInfos = []

	for aa in filenames_aa:
		schemaInfos.append(generateInfo_schema(aa))	
		unitInfos.append(generateInfo_unit(aa))

	units = sum(unitInfos,[])
	schemas = sum(schemaInfos,[])

	conflits = []

	for unit in units:
		for schema in schemas:
			if schema[0] == unit[0] and schema[3] != unit[3]: # [0]:id, [3]:polarity
				conflits.append(schema)
		
	return conflits

def getPosNegNeuTexts():
	filenames_aa = sorted(glob(op.join('.', 'aa1', '*.aa')))
	filenames_ac = sorted(glob(op.join('.', 'ac1', '*.ac')))

	unitInfos = []
	schemaInfos = []
	text_pos = []
	text_neg = []
	text_neu = []

	for i, aa in enumerate(filenames_aa):
		schemaInfos.append(generateInfo_schema(aa))	
		unitInfos.append(generateInfo_unit(aa))
		pol = setPolByLine(aa, filenames_ac[i], schemaInfos[i], unitInfos[i])
		text_pos.append(pol[0])
		text_neg.append(pol[1])
		text_neu.append(pol[2])
	return text_pos, text_neg, text_neu


# print existConflits()



