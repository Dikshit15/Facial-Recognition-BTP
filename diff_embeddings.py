import pickle
import face_recognition
import numpy as np
data = pickle.loads(open("output/embeddings.pickle", "rb").read())
# print(data['embeddings'])
# print(data['names'])

new_dict = {}
for i in data['names']:
	new_dict[i] =[]
for i in range(len(data['names'])):
	new_dict[data['names'][i]].append(data['embeddings'][i])

ref_embed = new_dict['dikshit'][0]
diff_record=[]
for i in new_dict:
	l = new_dict[i][0]
	diff= np.linalg.norm([ref_embed]-l, axis =1)
		# print i, j
		
	# print("hgfskahfdka")
	# print (diff)
	print("aniket " + i + "    " + str(diff))

for i in new_dict['dikshit']:
	diff= 0
	for a, b in zip(ref_embed, i):
		# print i, j
		diff= face_recognition.face_distance([ref_embed], l)
	print("aniket " + "aniket" + "    " + str(diff))



