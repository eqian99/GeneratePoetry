from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		'''
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		'''
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
print(generate_seq(model, mapping, 10, 'hello worl', 20))

line_num = 14
char_num = 40
chars_length = 26





poem = []
for i in range(line_num):
	string = chr(np.random.choice(list(range(chars_length))) + 97)
	for j in range(char_num):
		# goes here
		probs = model.predict(encoded, verbose=0)[0]
		yhat = np.random.choice(chars_length, 1, p=probs)
		char = chr(chars_length + 97)
		string += char
	poem.append(string)
