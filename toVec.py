from numpy import zeros
class onehot:
    __word_dict = dict()
    __size = int()

    def __init__(self, sentences):
        word_set = set()
        self.__size = 0
        for sentence in sentences:
            for word in sentence:
                word_set.add(word)

        for word in word_set:
            self.__word_dict[word] = self.__size
            self.__size += 1
    
    def __getitem__(self, word):
        rev = zeros(self.__size)
        for i in word:
            rev[self.__word_dict[i]] = 1
        return rev
    
    def __len__(self):
        return self.__size