from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'char_sequences2.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

bad_chars = sorted(list(set(raw_text)))
chars = list()
for char in bad_chars:
	if (char != '\n'):
		chars.append(char)
mapping = dict((c, i) for i, c in enumerate(chars))
revmapping = dict((i, c) for i, c in enumerate(chars))
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))
dump(revmapping, open('revmapping.pkl', 'wb'))

sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define temperatures
lambdas = [1.5, 0.75, 0.25]

# define model
model = Sequential()
model.add(LSTM(150, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
model.add(Lambda(lambda x: x * 0.25))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=500, verbose=2)

# save the model to file
model.save('model.h5')
'''
if (lam == 1.5):
	model.save('model1.h5')
elif (lam == 0.75):
	model.save('model2.h5')
else:
	model.save('model3.h5')
	'''
