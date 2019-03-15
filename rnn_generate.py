from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np

'''
# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers

		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])

		# predict character
		probs = model.predict(encoded, verbose=0)[0]
		yhat = np.random.choice(26, 1, p=probs)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping_converted.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20)) '''

# Constants
line_num = 14
char_num = 40
chars_length = 29
alphabet_start = 3

# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
revmapping = load(open('revmapping.pkl', 'rb'))
print(mapping)
print(len(mapping))

lambdas = [1.5, 0.75, 0.25]
lambdas = [0.25]

for lam in lambdas:
	# load the model
	'''
	if (lam == 1.5):
		model = load_model('model1.h5')
	if (lam == 0.75):
		model = load_model('model2.h5')
	if (lam == 0.25):
		model = load_model('model3.h5')
		'''
	model = load_model('model.h5')
	print("Temperature")
	print(lam)
	print('\n')
	poem = []
	for i in range(line_num):
		# string_index = np.random.choice(list(range(alphabet_start, chars_length)))
		# string = revmapping[string_index]
		if (i == 0):
			string = "shall i compare thee to a summer's day "
			new_string = string
		else:
			new_string = ""
			for j in range(char_num - 1):
				# encode the characters as integers
				encoded = [mapping[char] for char in string]
				# truncate sequences to a fixed length
				encoded = pad_sequences([encoded], maxlen=char_num, truncating='pre')
				# one hot encode
				encoded = to_categorical(encoded, num_classes=len(mapping))
				encoded = encoded.reshape(1, encoded.shape[1], encoded.shape[2])

				probs = model.predict(encoded, verbose=0)[0] / lam
				yhat = np.random.choice(chars_length, 1, p=probs)

				char = revmapping[yhat[0]]
				string += char
				new_string += char
		poem.append(new_string)
	for e in poem:
		print (e)
	print('\n')
