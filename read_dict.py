filepath = 'data/Syllable_dictionary.txt'
dict = {}
with open(filepath) as fp:
   for cnt, line in enumerate(fp):
       lst = line.split()
       dict[lst[0]] = lst[-1]

print(dict['anchored'])
