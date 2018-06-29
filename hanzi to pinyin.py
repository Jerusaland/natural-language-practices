from pypinyin import pinyin
from unidecode import unidecode
import pypinyin
import re
import json
import collections
import itertools

chars = [] # store all Chinese characters

with open("characters.txt", "r") as file:
	for line in file:
		try:
			chars.append(line.strip())
		except:
			pass

# map each character to its pinyin
l = list(map(lambda x: pinyin(x, heteronym=True,strict=True,style=pypinyin.NORMAL), chars))
# flatten list
l = list(itertools.chain(*list(itertools.chain(*l))))

# decode unicode into ascii
l = list(map(unidecode, l))
# remove duplicates
syllables = list(set(l))
# filter invalid pinyin
r = re.compile("[a-z]+")
syllables = list(filter(r.match, sorted(syllables)))

# build dictionary
d = {}
for i in syllables:
	start = i[0]
	if d.get(start) == None:
		d[start] = [i]
	else:
		d[start].append(i)
od = collections.OrderedDict(sorted(d.items()))

# write JSON representation
f = open("./result.json","w")
dump = json.dumps(od, indent=4)
print(dump)
f.write(dump)
f.close()

print(len(syllables))

