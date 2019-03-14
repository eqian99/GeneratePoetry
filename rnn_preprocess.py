def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

raw_text = load_doc('data/shakespeare.txt')

tokens = raw_text.split()
# remove numbers
tokens = [item for item in tokens if not item.isdigit()]
raw_text = ' '.join(tokens)
length = 40
sequences = list()
for i in range(length, len(raw_text)):
	seq = raw_text[i-length:i+1]
	sequences.append(seq)

out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)
