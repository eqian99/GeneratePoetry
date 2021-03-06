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

def load_word_list(path):
    """
    Loads a list of the words from the file at path <path>, removing all
    non-alpha-numeric characters from the file.
    """
    with open(path) as handle:
        # Load a list of whitespace-delimited words from the specified file
        raw_text = handle.read().strip().split()
        new_text = []
        for l in raw_text:
            new_text.append(''.join([i for i in l if not i.isdigit()]))
        # Strip non-alphanumeric characters from each word
        alphanumeric_words = map(lambda word: ''.join(char for char in word if char.isalnum()), new_text)
        # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
        alphanumeric_words = filter(lambda word: len(word) > 0, alphanumeric_words)

        # Convert each word to lowercase and return the result
        return list(map(lambda word: word.lower(), alphanumeric_words))

def tokenize(path):
    """
    Loads a list of the words from the file at path <path>, removing all
    non-alpha-numeric characters from the file.
    """
    with open(path) as handle:
        # Load a list of whitespace-delimited words from the specified file
        raw_text = handle.read().strip().splitlines()
        new_lines = []
        for l in raw_text:
            temp_l = l.replace('!', '').replace('?', '').replace(';', '').replace(',', '').replace(':', '').replace('.', '').lstrip()
            new_string = ''.join([i for i in temp_l if not i.isdigit()])
            new_lines.append(new_string)
#         # Strip non-alphanumeric characters from each word
#         alphanumeric_words = map(lambda word: ''.join(char for char in word if char.isalnum()), raw_text)
        # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
        alphanumeric_words = filter(lambda word: len(word) > 0, new_lines)
        # Convert each word to lowercase and return the result
        return list(map(lambda word: word.lower(), alphanumeric_words))

raw_text = load_doc('data/shakespeare.txt').lower()

tokens = raw_text.split()
raw_text = ' '.join(tokens).replace(')','').replace('(', '').replace('!', '').replace('?', '').replace(';', '').replace(',', '').replace(':', '').replace('.', '')
length = 40
sequences = list()
i = length
while i < len(raw_text):
    if raw_text[i].isdigit():
        i += (length + 1)
        if i >= len(raw_text):
            break
    if raw_text[i-length].isdigit():
        i += 2
        continue
    seq = raw_text[i-length:i+1]
    sequences.append(seq)
    i += 1

out_filename = 'char_sequences2.txt'
save_doc(sequences, out_filename)
